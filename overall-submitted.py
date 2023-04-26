import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
import time
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.cmf import CMF
from torchfm.model.clfm import CLFM
from torchfm.model.dtcdr import DTCDR
from torchfm.network import kmax_pooling
from selected_data.data_process import CrossDataset, SelectedDataset, save_selected_data, clear_selected_data


class ControllerNetwork_instance(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        self.controller_losses = None

    def forward(self, x):
        embed_x = self.embedding(x)
        output_layer = self.mlp(embed_x.view(-1, self.embed_output_dim))
        return output_layer

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 2))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

def set_random_seed(seed):
    print("* random_seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_dataset(name, path):
    if name == 'amazon':
        return SelectedDataset(path, name='amazon')
    elif name == 'ml25m':
        return SelectedDataset(path, name='ml25m')
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, field_dims):
    if name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(32,32,32), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(32,32,32), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(32,32,32), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'cmf':
        return CMF(field_dims, embed_dim=16, alpha=0.5, lamda=0.0, gamma=0.0)
    elif name == 'clfm':
        return CLFM(field_dims, user_embedding_size=32, item_embedding_size=16, share_embedding_size=8, alpha=0.5,
                    reg_weight=1e-4)
    elif name == 'dtcdr':
        return DTCDR(field_dims, embedding_size=16, mlp_hidden_size=[16, 16], dropout_prob=0.3, alpha=0.3,
                     base_model="NeuMF")
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_auc = 0
        self.logloss = 0.0
        self.save_path = save_path

    def is_continuable(self, model, accuracy, logloss):
        if accuracy > self.best_auc:
            self.best_auc = accuracy
            self.logloss = logloss
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train_target_one_epoch(model, optimizer, data_loader, criterion, device):
    model.train()
    for i, (fields, label, _) in enumerate(tqdm(data_loader)):
        fields, label = fields.long(), label.long()
        fields, label = fields.to(device), label.to(device)
        y = model(fields)
        loss_list = criterion(y, label.float())
        loss = loss_list.mean()
        model.zero_grad()
        loss.backward()
        optimizer.step()

def train_cdr_one_epoch(model, optimizer, data_loader, device, overlap_user=None):
    model.train()
    if overlap_user is not None:
        for i, (fields, label, cross) in enumerate(tqdm(data_loader)):
            fields, label, cross, overlap_user = fields.long().to(device), label.float().to(
                device), cross.long().to(device), overlap_user.long().to(device)
            loss = model.calculate_loss(fields, label, cross, overlap_user)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        return
    for i, (fields, label, cross) in enumerate(data_loader):
        fields, label, cross = fields.long(), label.float(), cross.long()
        fields, label, cross = fields.to(device), label.to(device), cross.to(device)
        loss = model.calculate_loss(fields, label, cross)
        model.zero_grad()
        loss.backward()
        optimizer.step()


def test_all(model, data_loader, device, overlap_user=None):
    model.eval()
    labels, predicts = [], []
    if overlap_user is not None:
        overlap_user = overlap_user.to(device)
        with torch.no_grad():
            for fields, label in data_loader:
                fields, label = fields.long().to(device), label.float().to(device)
                index = []
                noindex = []
                for j in range(fields.shape[0]):
                    if fields[j, 0] in overlap_user:
                        index.append(j)
                    else:
                        noindex.append(j)
                if len(index) != 0:
                    fields_overlap = fields[index]
                    label_overlap = label[index]
                    y_o = model(fields_overlap, overlap=True)
                    labels.extend(label_overlap.tolist())
                    predicts.extend(y_o.tolist())
                if len(noindex) != 0:
                    fields_no = fields[noindex]
                    label_no = label[noindex]
                    fields_no, label_no = fields_no.long().to(device), label_no.float().to(device)
                    y_no = model(fields_no)
                    labels.extend(label_no.tolist())
                    predicts.extend(y_no.tolist())
        return roc_auc_score(labels, predicts), log_loss(labels, predicts)
    with torch.no_grad():
        for fields, label in data_loader:
            fields, label = fields.to(device), label.to(device)
            y = model(fields)
            labels.extend(label.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(labels, predicts), log_loss(labels, predicts)


def test_a_batch(model, fields, label):
    model.eval()
    predicts = model(fields).tolist()
    labels = label.tolist()
    return roc_auc_score(labels, predicts), log_loss(labels, predicts)


def train_noFullBatch(model_name, field_dims, learning_rate, weight_decay, valid_data_loader, controller, optimizer_controller, data_loader, criterion, device, ControllerLoss, epsilon):
    model = []
    optimizer = []
    mra = 2
    threshold = torch.tensor(0.5).to(device)
    for x in range(mra):
        model.append(get_model(model_name, field_dims).to(device))
        optimizer.append(torch.optim.Adam(
            params=model[x].parameters(), lr=learning_rate, weight_decay=weight_decay))
    for i, (fields, label, cross) in enumerate(tqdm(data_loader)):
        fields, label, cross = fields.to(device), label.float().to(device), cross.to(device)
        for model_x in model:
            model_x.eval()
        controller.eval()
        has_target = True
        has_source = True
        with torch.no_grad():
            # seperate target and source
            label = label.reshape((-1, 1))
            target_idx = torch.nonzero(cross).squeeze()
            if target_idx is None or target_idx.dim() == 0 or target_idx.shape[0]== 0:
                has_target = False
                print("Target is none.")
            else:
                target_idx = target_idx.reshape((-1, 1))
                target_fields = torch.gather(
                    fields, 0, target_idx.repeat(1, fields.shape[1]))
                target_label = (torch.gather(
                    label, 0, target_idx.repeat(1, label.shape[1]))).squeeze()

            source_idx = torch.nonzero(1 - cross).squeeze()
            if source_idx is None or source_idx.dim() == 0 or source_idx.shape[0]== 0:
                has_source = False
                print("Source is none.")
            else:
                source_idx = source_idx.reshape((-1, 1))
                source_fields = torch.gather(
                    fields, 0, source_idx.repeat(1, fields.shape[1]))
                source_label = torch.gather(
                    label, 0, source_idx.repeat(1, label.shape[1])).squeeze()

            # validation before update
            auc = 0
            test_fields, test_label = next(iter(valid_data_loader))
            test_fields, test_label = test_fields.to(device), test_label.to(device)
            for x in range(mra):
                auc_x,_ = test_a_batch(model[x], test_fields, test_label)
                auc+=auc_x
            auc/=mra

        controller.train()
        output_layer = controller(source_fields)
        output_layer1 = output_layer.detach()
        # sample the action, calculate the loss before update
        with torch.no_grad():
            # threshold = torch.tensor(min(random.random(), 0.9)).to(device)
            prob_instance = torch.softmax(output_layer1, dim=-1)

            sampled_actions = torch.where(prob_instance[:, 1] > threshold, 1, 0)
            sampled_actions = torch.tensor(
                [action if random.random() >= epsilon else -(action - 1) for action in sampled_actions]).to(device)
            prob_idx = torch.nonzero(sampled_actions).squeeze()
            if prob_idx is None or prob_idx.dim()==0 or prob_idx.shape[0]== 0:
                print("No instance sampled.")
                continue
            selected_label = torch.gather(source_label, 0, prob_idx)
            selected_instance = torch.gather(
                source_fields, 0, prob_idx.reshape((-1,1)).repeat(1, source_fields.shape[1]))
            loss_list = None
            for x in range(mra):
                y = model[x](source_fields)
                loss_list = criterion(y, source_label).reshape((1,-1)).float() if loss_list is None else torch.cat((loss_list, criterion(y, source_label).reshape((1,-1)).float()))
            loss_list = torch.mean(loss_list,dim=0)

        # update RS model
        for model_x in model:
            model_x.train()
        for x in range(mra):
            if not has_target:
                y_sl = model[x](selected_instance)
                loss_list_sl = criterion(y_sl, selected_label)
                loss = loss_list_sl.mean()
            elif not has_source:
                y_target = model[x](target_fields)
                loss_list_target = criterion(y_target, target_label)
                loss = loss_list_target.mean()
            else:
                fields = torch.cat((target_fields, selected_instance))
                label = torch.cat((target_label, selected_label))
                y = model[x](fields)
                loss_list_all = criterion(y, label)
                loss = loss_list_all.mean()

            model[x].zero_grad()
            loss.backward()
            optimizer[x].step()
            model[x].eval()

        # update optimizer
        if has_source:
            # validation after update
            with torch.no_grad():
                auc1 = 0
                for x in range(mra):
                    auc_x, _ = test_a_batch(model[x], test_fields, test_label)
                    auc1 += auc_x
                auc1 /= mra
                # calculate the loss after update
                loss_list1 = None
                for x in range(mra):
                    y = model[x](source_fields)
                    loss_list1 = criterion(y, source_label).reshape((1,-1)).float() if loss_list1 is None else torch.cat(
                        (loss_list1, criterion(y, source_label).reshape((1,-1)).float()))
                loss_list1 = torch.mean(loss_list1, dim=0)
                # calculate reward
                reward = (auc1-auc) * (loss_list - loss_list1)
            c_loss = torch.sum(ControllerLoss(output_layer, sampled_actions) * reward)

            controller.zero_grad()
            c_loss.backward()
            optimizer_controller.step()

    print("Probability: {}".format(prob_instance[:5, 1]))



def test(model, data_loader, device):
    model.eval()
    labels, predicts = list(), list()
    with torch.no_grad():
        for fields, label in data_loader:
            fields, label = fields.to(device), label.to(device)
            y = model(fields)
            labels.extend(label.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(labels, predicts), log_loss(labels, predicts)


def train_target(field_dims,train_data_loader, valid_data_loader, test_data_loader,
             model_name, learning_rate, criterion, weight_decay, device, save_rs_name):
    model = get_model(model_name, field_dims).to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(
        num_trials=3, save_path=save_rs_name)
    train_start_time = time.time()

    for epoch_i in range(40):
        train_target_one_epoch(model, optimizer, train_data_loader,
                  criterion, device)
        auc, logloss = test_all(model, valid_data_loader, device)
        print('\tepoch:', epoch_i,
              'validation: auc:', auc, 'logloss:', logloss)
        if not early_stopper.is_continuable(model, auc, logloss):
            print(f'\tvalidation: best auc: {early_stopper.best_auc}')
            break
    train_end_time = time.time()
    print("\tTime of target training: {:.2f}min".format(
        (train_end_time - train_start_time)/60))
    model.load_state_dict(torch.load(save_rs_name))
    if test_data_loader is not None:
        auc, logloss = test_all(model, test_data_loader, device)
        print(f'\ttest auc: {auc}, logloss: {logloss}\n')
        return auc, logloss, model
    return early_stopper.best_auc, early_stopper.logloss, model

def train_cdr(field_dims, train_data_loader, valid_data_loader, test_data_loader,
             model_name, learning_rate, weight_decay, device, save_cdr_name, epoch, overlap_user=None):
    model = get_model(model_name, field_dims).to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(
        num_trials=3, save_path=save_cdr_name)
    train_start_time = time.time()
    for epoch_i in range(epoch):
        train_cdr_one_epoch(model, optimizer, train_data_loader, device, overlap_user)
        auc, logloss = test_all(model, valid_data_loader, device, overlap_user)
        print('\tepoch:', epoch_i,
              'validation: auc:', auc, 'logloss:', logloss)
        if not early_stopper.is_continuable(model, auc, logloss):
            print(f'\tvalidation: best auc: {early_stopper.best_auc}')
            break
    train_end_time = time.time()
    print("\tTime of training: {:.2f}min".format(
        (train_end_time - train_start_time)/60))
    model.load_state_dict(torch.load(save_cdr_name))
    if test_data_loader is not None:
        auc, logloss = test_all(model, test_data_loader, device)
        print(f'\ttest auc: {auc}, logloss: {logloss}\n')
        return auc, logloss, model
    return early_stopper.best_auc, early_stopper.logloss, model


def save_test_validset(data_loader, selected_data_path):
    clear_selected_data(selected_data_path)
    print('Start saving', selected_data_path)
    for (fields, label) in data_loader:
        save_selected_data(selected_data_path, fields.cpu(
        ).numpy().copy(), label.cpu().numpy().copy())
    print('Finish saving.')

def select_instance(data_set, selected_data_path, controller, device, select_ratio):
    clear_selected_data(selected_data_path)
    slct_number = int(select_ratio*len(data_set))
    fields, label = torch.from_numpy(data_set.field).to(device), torch.from_numpy(data_set.label).to(device)
    controller.eval()
    output_layer = controller(fields)
    prob_instance = torch.softmax(output_layer, dim=-1)
    prob_idx = kmax_pooling(prob_instance[:,1], 0, slct_number)
    if prob_idx is not None and prob_idx.dim != 0:
        if prob_idx.size(dim=0) != 0:
            # print("Probability: ", prob_instance[prob_idx[:5], 1], prob_instance[prob_idx[-5:], 1])
            selected_label = torch.gather(label, 0, prob_idx)
            selected_instance = torch.gather(
                fields, 0, prob_idx.unsqueeze(1).repeat(1, fields.shape[1]))
            save_selected_data(selected_data_path, selected_instance.cpu().numpy(), selected_label.cpu().numpy())
        else:
            print("No sample selected!")
    else:
        print("No sample selected!")
    print("---------------------------------Select_ratio: ", select_ratio)
    print("---------------------------------Select: ", len(selected_label))
    return slct_number

def main(dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         select_ratio,
         epsilon,
         dataset_setting,
         ratio_num,
         select_start,
         select_end):

    if dataset_setting=="amazon":
        #0.9
        trans_name = 'Automotive'
        #0.17
        # trans_name = 'Toys_and_Games'
        dataset_name = 'Industrial_and_Scientific'
    elif dataset_setting>"ml25m":
        trans_name = 'ml-25m_11'
        dataset_name = 'ml-25m_10'
    else:
        raise ValueError('unknown dataset setting: ' + dataset_setting)


    print('Dataset setting:', dataset_setting)
    print('Dataset path:', dataset_path)
    path_source = './data/' + trans_name + '/' + trans_name + '.inter'
    path_target = './data/' + dataset_name + '/' + dataset_name + '.inter'
    print("-source dataset:", path_source)
    print("-target dataset:", path_target)
    print('Save dir:', save_dir)

    print('Model name:', model_name)
    print('Epoch:', epoch)
    print('Learning rate:', learning_rate)
    print('Batch size:', batch_size)
    print('Weight decay:', weight_decay)
    print('Device:', device)
    print('Select ratio:', select_ratio)
    print('Epsilon:', epsilon)
    print('Ratio num:', ratio_num)
    print('Select start:', select_start)
    print('Select end:', select_end)

    device = torch.device(device)
    dataset_trans = get_dataset(dataset_setting, path_source)
    dataset_target = get_dataset(dataset_setting, path_target)
    dataset_train, dataset_valid, dataset_test = dataset_target.split()

    info = '{}_{}_{}_{}_{}'.format(model_name, dataset_setting, str(epoch), str(batch_size), str(learning_rate))
    save_controller_name = './{}/controller_whole_'.format(
        save_dir) + info + '.pt'
    save_rs_name = save_controller_name.replace('controller', 'rs')
    selected_data_path = './selected_data/notFixed_whole_{}_{}_train.txt'.format(
        model_name, dataset_setting)

    if model_name in ["cmf", "clfm", "dtcdr"]:
        print('Training batch size:', batch_size)
        save_cdr_name = './{}/cdr_whole_'.format(
            save_dir) + info + '.pt'

        dataset_train = CrossDataset(dataset_train, dataset_trans)

        overlap_user = None
        if model_name == 'dtcdr':
            overlap_user = CrossDataset(dataset_target, dataset_trans).overlap_user
            
        field_dims = []
        for i in range(len(dataset_train.field_dims)):
            field_dims.append(max(dataset_train.field_dims[i], dataset_valid.field_dims[i],
                                  dataset_test.field_dims[i]))

        train_data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        valid_data_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
            
        train_cdr(field_dims, train_data_loader, valid_data_loader, test_data_loader,
                     model_name, learning_rate, weight_decay, device, save_cdr_name, epoch, overlap_user)
        return
    
    criterion = torch.nn.BCELoss(reduction='none')

    dataset_cross = CrossDataset(dataset_train, dataset_trans)
    dataset_train = CrossDataset(dataset_train, None)
    train_data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    cross_data_loader = DataLoader(dataset_cross, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    field_dims = []
    for i in range(len(dataset_train.field_dims)):
        field_dims.append(max(dataset_cross.field_dims[i], dataset_valid.field_dims[i],
                              dataset_test.field_dims[i]))


    controller = ControllerNetwork_instance(
        field_dims, embed_dim=16, mlp_dims=(64, 64), dropout=0.2).to(device)
    optimizer_controller = torch.optim.Adam(
        params=controller.parameters(), lr=learning_rate*0.001, weight_decay=weight_decay)
    ControllerLoss = nn.CrossEntropyLoss(reduction='none')


    print('\n****************************** Searching Phase ******************************\n')
    trans_num = len(dataset_trans)
    for epoch_i in range(epoch):
        print("\n****************************** Epoch: {} *********************************".format(epoch_i))
        train_noFullBatch(model_name, field_dims, learning_rate, weight_decay, valid_data_loader, controller, optimizer_controller, cross_data_loader,
                              criterion, device, ControllerLoss, epsilon)
    torch.save(controller.state_dict(), save_controller_name)

    print(
        '\n\t========================= Target Training Phase ==========================\n')

    print('\t# start selection...')
    max_auc = 0.0
    max_logloss = 1000000.0
    test_auc = 0
    test_logloss = 0
    max_ratio = 0.0
    max_slct = 0
    auc_list = [0]
    logloss_list = [0]
    x_list = [0.0]
    if select_ratio is None:
        for select_ratio in np.linspace(select_start, select_end, num=ratio_num):
            x_list.append(select_ratio)

            slct_number = select_instance(dataset_trans, selected_data_path, controller, device,
                                          select_ratio)
            dataset_select = SelectedDataset(selected_data_path, name="selected")
            print("\tRatio: {}\tSelect number: [{} / {}]\tSelect path: {}".format(select_ratio, slct_number, trans_num, selected_data_path))
            dataset_merge = CrossDataset(dataset_train, dataset_select)
            selected_data_loader = DataLoader(dataset_merge, batch_size=batch_size, shuffle=True)
            retrain_auc, retrain_logloss, model = train_target(field_dims, selected_data_loader, valid_data_loader,
                                                           None, model_name, learning_rate, criterion, weight_decay, device,
                                                           save_rs_name)
            # true test auc and logloss for reference, but not for training use
            auc, logloss = test_all(model, test_data_loader, device)

            auc_list.append(auc)
            logloss_list.append(logloss)

            if retrain_auc>max_auc:
                max_ratio = select_ratio
                max_auc=retrain_auc
                max_logloss = retrain_logloss
                max_slct = slct_number
                test_auc = auc
                test_logloss = logloss
    else:
        x_list.append(select_ratio)

        slct_number = select_instance(dataset_trans, selected_data_path, controller, device,
                                      select_ratio)
        dataset_select = SelectedDataset(selected_data_path, name="selected")
        print("\tRatio: {}\tSelect number: [{} / {}]\tSelect path: {}".format(select_ratio, slct_number, trans_num,
                                                                              selected_data_path))
        dataset_merge = CrossDataset(dataset_train, dataset_select)
        selected_data_loader = DataLoader(dataset_merge, batch_size=batch_size, shuffle=True)
        retrain_auc, retrain_logloss, model = train_target(field_dims, selected_data_loader, valid_data_loader,
                                                           None, model_name, learning_rate, criterion, weight_decay,
                                                           device,
                                                           save_rs_name)
        # true test auc and logloss for reference, but not for training use
        auc, logloss = test_all(model, test_data_loader, device)

        auc_list.append(auc)
        logloss_list.append(logloss)

        if retrain_auc > max_auc:
            max_ratio = select_ratio
            max_auc = retrain_auc
            max_logloss = retrain_logloss
            max_slct = slct_number
            test_auc = auc
            test_logloss = logloss

    # training with target set only (ratio=0.0)
    print("\tTesting Target...")
    auc_list[0], logloss_list[0], _ = train_target(field_dims, train_data_loader, valid_data_loader,
                                             test_data_loader,
                                             model_name, learning_rate, criterion, weight_decay, device,
                                             save_rs_name)

    print("\tTesting Merge...")
    dataset_merge = CrossDataset(dataset_train, dataset_trans)
    merge_data_loader = DataLoader(dataset_merge, batch_size=batch_size, shuffle=True)
    auc_merge, logloss_merge, _ = train_target(field_dims, merge_data_loader, valid_data_loader,
                                                   test_data_loader,
                                                   model_name, learning_rate, criterion, weight_decay, device,
                                                   save_rs_name)
    auc_list.append(auc_merge)
    logloss_list.append(logloss_merge)
    x_list.append(1.0)
    print('\t========================= End Training and Testing =========================\n')
    print("\t----------------------------------------------------------------------------\n")

    print("Target: auc_{:.8f}  logloss_{:.8f}".format(
        auc_list[0], logloss_list[0]))
    print("Merge: auc_{:.8f}  logloss_{:.8f}".format(
        auc_list[-1], logloss_list[-1]))
    print("Selection: auc_{:.8f}  logloss_{:.8f}".format(
        test_auc, test_logloss))

    print("Best Ratio: {}, Slct: [{} / {}]".format(max_ratio, max_slct, trans_num))
    final_performance=[max_auc,max_logloss]
    with open('Record_data/%s_%s_notFixed_whole.txt' % (model_name, dataset_setting), 'a') as the_file:
        the_file.write('\nModel:%s\nDataset:%s\nSearching Epoches: %d\nLearning Rate: %s\nBatch Size: %d\nWeight Decay: %s\nDevice: %s\nSelect Ratio: %s\nEpsilon: %s\nRatio Num: %d\nSelect start: %s\nSelect End: %s\nFinal performance: %s\n'
                   % (model_name, dataset_setting, epoch_i+1, str(learning_rate), batch_size, str(weight_decay), device, str(max_ratio), str(epsilon), ratio_num, str(select_start), str(select_end), str(final_performance)))
    print("Ratio x_axis: ", x_list)
    print("AUC axis: ", auc_list)
    print("Logloss axis: ", logloss_list)
    print("Selected data save to: ", selected_data_path)
    print("controller save to: ", save_controller_name)
    print("Best rs model save to: ", save_rs_name)




if __name__ == '__main__':
    # set_random_seed(56789)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_setting', default="amazon")
    parser.add_argument(
        '--dataset_path', default='')
    parser.add_argument('--model_name', default='afm')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--ratio_num', type=int, default=20)
    parser.add_argument('--select_start', type=float, default=0.05)
    parser.add_argument('--select_end', type=float, default=0.95)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument(
        '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default='save_model')
    parser.add_argument('--repeat_experiments', type=int, default=1)
    parser.add_argument('--select_ratio',  type=float, default=None)
    parser.add_argument('--epsilon', type=float, default=0.1)
    args = parser.parse_args()


    for i in range(args.repeat_experiments):
        main(args.dataset_path,
             args.model_name,
             args.epoch,
             args.learning_rate,
             args.batch_size,
             args.weight_decay,
             args.device,
             args.save_dir,
             args.select_ratio,
             args.epsilon,
             args.dataset_setting,
             args.ratio_num,
             args.select_start,
             args.select_end)
