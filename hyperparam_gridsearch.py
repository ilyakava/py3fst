"""Iterate through hyperparams to get accuracies
"""

import os
import time

import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm

import hyper_pixelNN as hsinn

import pdb

def run_svm(trainX, trainY, testX, testY):
    print('starting training')
    start = time.time()
    clf = SVC(kernel='linear')
    clf.fit(trainX, trainY)
    end = time.time()
    print('Training done. Took %is' % int(end - start))

    # now test
    test_chunk_size = 1000

    n_correct = 0
    for i in tqdm(range(0,testY.shape[0],test_chunk_size), desc='Testing'):
        p_label = clf.predict(testX[i:i+test_chunk_size]);
        n_correct += (p_label == testY[i:i+test_chunk_size]).sum()
    acc = float(n_correct) / testY.shape[0]
    return acc
        

def get_acc_for_config(dataset, st_net_spec, preprocessed_data_path, preprocessed_data_root, mask_paths, save_features):
    """One set of hyperparams and dataset, several trials
    """
    trainimgname, trainlabelname = hsinn.dset_filenames_dict[dataset]
    trainimgfield, trainlabelfield = hsinn.dset_fieldnames_dict[dataset]
    data, labels = hsinn.load_data(trainimgname, trainimgfield, trainlabelname, trainlabelfield)
    
    if save_features:
        data = hsinn.load_or_preprocess_data(data, preprocessed_data_path, preprocessed_data_root, st_net_spec, 31)
    else:
        data = hsinn.preprocess_data(data, st_net_spec, 31)
    
    accs = []
    # loop through masks
    for mask_path in mask_paths:
        train_mask = hsinn.multiversion_matfile_get_field(mask_path, 'train_mask')
        val_mask = hsinn.multiversion_matfile_get_field(mask_path, 'test_mask')
        trainX, trainY, valX, valY = hsinn.get_train_val_splits(data, labels, train_mask, val_mask, (0,0,0)) # , n_eval=2048
    
        # run svm
        accs.append(run_svm(trainX.squeeze(), trainY, valX.squeeze(), valY))
    print('[%s] Avg acc is %.2f' % (hsinn.spec_to_str(st_net_spec), sum(accs) / len(accs)))
    return accs
    

def specs1():
    """
    Spatial psis should be decreasing, phi should be equal to smallest psi
    """
    specs = []
    myrange = [9,7,5,3]
    for b in myrange:
        for s1 in myrange:
            for s2 in myrange:
                if s1 >= s2:
                    specs.append( [ [b,s1,s1], [b,s2,s2], [b,s2,s2] ] )
    return specs
    
def specs_append9():
    """
    Spatial psis should be decreasing, phi should be equal to smallest psi
    """
    specs = []
    myrange = [9,7,5,3]
    for b in myrange:
        for s1 in [9]:
            for s2 in myrange:
                if s1 >= s2:
                    specs.append( [ [b,s1,s1], [b,s2,s2], [b,s2,s2] ] )
    return specs
    
def specs_append9_bands():
    """
    Spatial psis should be decreasing, phi should be equal to smallest psi
    """
    specs = []
    myrange = [7,5,3]
    for b in [9]:
        for s1 in myrange:
            for s2 in myrange:
                if s1 >= s2:
                    specs.append( [ [b,s1,s1], [b,s2,s2], [b,s2,s2] ] )
    return specs
    
def perform_gridsearch(dataset, masks, specs, outfile='gridsearch.npz', save_features=False, preprocessed_data_root='/scratch0/ilya/locDoc/data/hyperspec/features/npz_feat'):
    # a list< list < int > > -> list< float >.
    # Maps a configuration to a set of accuracies
    results = {}
    for spec_i, spec in enumerate(specs):
        psi1,psi2,phi = spec
    
        st_net_spec = hsinn.st_net_spec_struct(psi1,psi2,phi)
    
        preprocessed_data_path = os.path.join(preprocessed_data_root, '%s__%s.npz' % (dataset, hsinn.spec_to_str(st_net_spec)))
    
        accs = get_acc_for_config(dataset, st_net_spec, preprocessed_data_path, preprocessed_data_root, masks, save_features)
        results[hsinn.spec_to_str(st_net_spec)] = accs
        print('FINISHED %i/%i' % (spec_i+1, len(specs)))
    
        np.savez(os.path.join('/scratch0/ilya/locDoc/pyfst', outfile), results=results)
    print('Saved %s' % os.path.join('/scratch0/ilya/locDoc/pyfst', outfile))
    

def paviaU():
    dataset = 'PaviaU'
    # masks = ['PaviaU_gt_traintest_s03_1_3f6384.mat',
    #          'PaviaU_gt_traintest_s03_2_b67b5f.mat',
    #          'PaviaU_gt_traintest_s03_3_7d8356.mat',
    #          'PaviaU_gt_traintest_s03_4_241266.mat',
    #         'PaviaU_gt_traintest_s03_5_ccbbb1.mat',
    #         'PaviaU_gt_traintest_s03_6_dce186.mat',
    #         'PaviaU_gt_traintest_s03_7_d5cdfe.mat',
    #         'PaviaU_gt_traintest_s03_8_6bcd5a.mat',
    #         'PaviaU_gt_traintest_s03_9_a1ff2b.mat',
    #         'PaviaU_gt_traintest_s03_10_e1dac2.mat']
    
             
    masks = ['PaviaU_gt_traintest_s200_10_149f64.mat',
        'PaviaU_gt_traintest_s200_1_591636.mat',
        'PaviaU_gt_traintest_s200_2_2255d5.mat',
        'PaviaU_gt_traintest_s200_3_628d0a.mat',
        'PaviaU_gt_traintest_s200_4_26eddf.mat',
        'PaviaU_gt_traintest_s200_5_25dd01.mat',
        'PaviaU_gt_traintest_s200_6_2430e7.mat',
        'PaviaU_gt_traintest_s200_7_409d67.mat',
        'PaviaU_gt_traintest_s200_8_f79373.mat',
        'PaviaU_gt_traintest_s200_9_dac1e4.mat']
    
    masks = [os.path.join('/scratch0/ilya/locDoc/data/hyperspec/',m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_paviaU_s200_10trials_full_new.npz', save_features=True)
    masks = ['PaviaU_strictsinglesite_trainval_s20_0_276673.mat',
'PaviaU_strictsinglesite_trainval_s20_1_795837.mat',
'PaviaU_strictsinglesite_trainval_s20_2_161741.mat',
'PaviaU_strictsinglesite_trainval_s20_3_814061.mat',
'PaviaU_strictsinglesite_trainval_s20_4_185963.mat',
'PaviaU_strictsinglesite_trainval_s20_5_950681.mat',
'PaviaU_strictsinglesite_trainval_s20_6_578647.mat',
'PaviaU_strictsinglesite_trainval_s20_7_081381.mat',
'PaviaU_strictsinglesite_trainval_s20_8_124503.mat',
'PaviaU_strictsinglesite_trainval_s20_9_310413.mat']
    masks = [os.path.join('/scratch0/ilya/locDoc/data/hyperspec/',m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_paviaU_strictsingle_site_s20_10trials.npz', save_features=True)

def Botswana():
    dataset = 'Botswana'
    masks = ['Botswana_singlesite_trainval_s03_0_431593.mat',
            'Botswana_singlesite_trainval_s03_1_422869.mat',
            'Botswana_singlesite_trainval_s03_2_942165.mat',
            'Botswana_singlesite_trainval_s03_3_066225.mat',
            'Botswana_singlesite_trainval_s03_4_842055.mat',
            'Botswana_singlesite_trainval_s03_5_256397.mat',
            'Botswana_singlesite_trainval_s03_6_976203.mat',
            'Botswana_singlesite_trainval_s03_7_784583.mat',
            'Botswana_singlesite_trainval_s03_8_588663.mat',
            'Botswana_singlesite_trainval_s03_9_484915.mat']

    masks = [os.path.join('/scratch0/ilya/locDoc/data/hyperspec/',m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_Bots_singlesite_s03_10trials_new.npz')

def KSC():
    dataset = 'KSC'
    masks = ['KSC_strictsinglesite_trainval_s20_0_132440.mat',
        'KSC_strictsinglesite_trainval_s20_1_756652.mat',
        'KSC_strictsinglesite_trainval_s20_2_250680.mat',
        'KSC_strictsinglesite_trainval_s20_3_119240.mat',
        'KSC_strictsinglesite_trainval_s20_4_767740.mat',
        'KSC_strictsinglesite_trainval_s20_5_192528.mat',
        'KSC_strictsinglesite_trainval_s20_6_563040.mat',
        'KSC_strictsinglesite_trainval_s20_7_680204.mat',
        'KSC_strictsinglesite_trainval_s20_8_611960.mat',
        'KSC_strictsinglesite_trainval_s20_9_596964.mat']

    masks = [os.path.join('/scratch0/ilya/locDoc/data/hyperspec/', m) for m in masks]
    perform_gridsearch(dataset, masks, specs1(), outfile='gridsearch_KSC_strictsinglesite_s20_10trials.npz', save_features=True)


def main():
    paviaU()
    # Botswana()
    # KSC()

if __name__ == '__main__':
    main()