"""
Double DQN (Nature 2015)
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf

Notes:
    The difference is that now there are two DQNs (DQN & Target DQN)

    y_i = r_i + 𝛾 * max(Q(next_state, action; 𝜃_target))

    Loss: (y_i - Q(state, action; 𝜃))^2

    Every C step, 𝜃_target <- 𝜃

"""
import numpy as np
import tensorflow as tf
import random
from collections import deque
import dqn_cnn_iteration_till_end
import h5py
import time
import math as m
import scipy.io
from numpy import *
import scipy.linalg
import matplotlib.pyplot as plt
import os
import pylab as pl
from tensorflow.python.framework import dtypes
from typing import List


#env = gym.make('CartPole-v0')
#env = gym.wrappers.Monitor(env, directory="gym-results/", force=True)

# Constants defining our neural network
INPUT_SIZE = 9
PATCH_reward = 5
OUTPUT_SIZE = 5
NPROJ = 180
NP = 192
TRAIN_IMG_NUM = 6
TEST_IMG_NUM = 6
MAXITER_RECON =30
NPixel = 128
PATCH_NUM = NPixel*NPixel
Train_NUM_total = PATCH_NUM*TRAIN_IMG_NUM
DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 5000000
BATCH_SIZE = 128
TARGET_UPDATE_FREQUENCY = 15
MAX_EPISODES = 300
load_session=0
save_session=1

TRAIN_NUM_ITER=10


def replay_train(mainDQN: dqn_cnn_iteration_till_end.DQN, targetDQN: dqn_cnn_iteration_till_end.DQN, states, next_states, actions, rewards,done,para) -> float:
    X = states
    X1 = next_states

    temp = np.max(targetDQN.predict(X1), axis=3)
    Q_target = rewards + DISCOUNT_RATE * temp[:,0,0]

    y = mainDQN.predict(X)
    for i in range(y.shape[0]):
        y[i,0,0,int(actions[i])] = Q_target[i]

    return mainDQN.update(X, y)


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def backupNetwork(model, backup):
    weightMatrix = []
    for layer in model.layers:
        weights = layer.get_weights()
        weightMatrix.append(weights)
    i = 0
    for layer in backup.layers:
        weights = weightMatrix[i]
        layer.set_weights(weights)
        i += 1

def grad(f, NPixel):
    fimg = np.reshape(f,(NPixel,NPixel),order='F')
    fimgx = zeros((NPixel,NPixel))
    fimgx[0:NPixel-1,:] = fimg[1:NPixel,:]
    fimgx[NPixel-1,:] = fimg[NPixel-1,:]
    fimgy = zeros((NPixel, NPixel))
    fimgy[:,0:NPixel-1] = fimg[:,1:NPixel]
    fimgy[:,NPixel-1] = fimg[:,NPixel-1]
    gradfimgx = fimgx-fimg
    gradfimgy = fimgy-fimg
    gradfx = np.reshape(gradfimgx,(NPixel*NPixel),order='F')
    gradfy = np.reshape(gradfimgy,(NPixel*NPixel), order='F')
    gradf = zeros((NPixel*NPixel,2))
    gradf[:,0] = gradfx
    gradf[:,1] = gradfy
    return gradf


def div(f,NPixel):
    gradx = f[:,0]
    grady = f[:,1]
    gradximg = np.reshape(gradx,(NPixel,NPixel),order='F')
    gradyimg = np.reshape(grady, (NPixel, NPixel), order='F')
    gradxximg = zeros((NPixel,NPixel))
    gradyyimg = zeros((NPixel, NPixel))
    gradxximg[1:NPixel,:] = gradximg[0:NPixel-1,:]
    gradxximg[NPixel-1, :] = gradxximg[NPixel-1, :] + gradximg[NPixel-2,:]
    gradyyimg[:,1:NPixel] = gradyimg[:,0:NPixel-1]
    gradyyimg[:,NPixel-1] = gradyyimg[:,NPixel-1]+ gradyimg[:,NPixel-2]
    divfimg = gradximg-gradxximg+gradyimg-gradyyimg
    divf = np.reshape(divfimg,(NPixel*NPixel,1),order='F')

    return divf


def laplacian(f,NPixel):
    gradf = grad(f,NPixel)
    lapf = -div(gradf,NPixel)
    return lapf



def reconTV(pMat,projdata,state, action, para, gamma,GroundTruth,NPixel,INPUT_SIZE,itertotal,tol):
    projdata = np.reshape(projdata,(NPROJ*NP,1),order='F')
    f = state[:,int((INPUT_SIZE*INPUT_SIZE+1)/2)-1]
    f = np.reshape(f,(PATCH_NUM,1),order='F')
    f0 = f
    for idx in range(PATCH_NUM):
        if action[idx]==0:
            para[idx]=para[idx]*1.5
            if para[idx]>10:
                para[idx]=10

        if action[idx] == 1:
            para[idx] = para[idx] * 1.1
            if para[idx]>10:
                para[idx]=10
        if action[idx] == 3:
            para[idx] = para[idx] * 0.9
            if para[idx]<0.00001:
                para[idx]=0.00001
        if action[idx]==4:
            para[idx]=para[idx]*0.5
            if para[idx]<0.00001:
                para[idx]=0.00001


    for IterOut in range(itertotal):
        gradf = grad(f, NPixel)
        gtemp = gradf + gamma / 2
        sgtemp = np.sign(gtemp)
        para2 = zeros((PATCH_NUM, 2))
        para2[:, 0] = para
        para2[:, 1] = para
        gtemp = np.absolute(gtemp) - para2 / 2
        for i in range(PATCH_NUM):
            for j in range(2):
                if gtemp[i, j] < 0:
                    gtemp[i, j] = 0

        g = np.multiply(sgtemp, gtemp)
        gamma = gamma + 1 * (gradf - g)
        fold = f

        rhs = pMat.transpose() * projdata + div(gamma, NPixel) - 2 * div(g, NPixel)
        temp = pMat * f
        temp = pMat.transpose() * temp
        lhs = temp + 2 * laplacian(f, NPixel)
        r = rhs - lhs
        p = r
        rsold = np.matmul(r.transpose(), r)
        for iterCG in range(5):

            tempp = pMat * p
            tempp = pMat.transpose() * tempp
            Ap = tempp + 2 * laplacian(p, NPixel)

            pAp = np.matmul(p.transpose(), Ap)
            alpha = rsold / pAp
            f = f + alpha * p
            for ind in range(PATCH_NUM):
                if f[ind] < 0:
                    f[ind] = 0

            r = r - alpha * Ap
            rsnew = np.matmul(r.transpose(), r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        if np.sum(np.absolute(f-fold))/np.sum(np.absolute(fold))<=tol:

            break
    print(IterOut)
    fimg = np.reshape(f, (NPixel, NPixel), order='F')
    fimgpad = zeros((NPixel+INPUT_SIZE-1,NPixel+INPUT_SIZE-1))
    fimgpad[int((INPUT_SIZE+1)/2)-1:NPixel+int((INPUT_SIZE+1)/2)-1,int((INPUT_SIZE+1)/2)-1:NPixel+int((INPUT_SIZE+1)/2)-1]=fimg
    next_state = zeros((PATCH_NUM,INPUT_SIZE*INPUT_SIZE))
    count = 0
    for xcord in range(NPixel):
        for ycord in range(NPixel):
            temp=fimgpad[ycord:ycord+INPUT_SIZE,xcord:xcord+INPUT_SIZE]
            next_state[count,:] = np.reshape(temp,(INPUT_SIZE*INPUT_SIZE),order='F')
            count += 1

    dist1 = reshape(f0,(PATCH_NUM),order='F')-GroundTruth
    dist2 = reshape(f,(PATCH_NUM),order='F')-GroundTruth

    dist1img = np.reshape(dist1,(NPixel,NPixel),order = 'F')
    dist2img = np.reshape(dist2, (NPixel, NPixel), order='F')
    dist1imgLarge = zeros((NPixel+PATCH_reward-1,NPixel+PATCH_reward-1))
    margin = int((PATCH_reward-1)/2)
    dist1imgLarge[margin:NPixel+margin,margin:NPixel+margin]=np.absolute(dist1img)

    dist2imgLarge = zeros((NPixel + PATCH_reward-1, NPixel + PATCH_reward-1))
    dist2imgLarge[margin:NPixel + margin, margin:NPixel + margin] = np.absolute(dist2img)

    GTimgLarge = zeros((NPixel + PATCH_reward-1, NPixel + PATCH_reward-1))
    GTimgLarge[margin:NPixel + margin, margin:NPixel + margin] = reshape(GroundTruth,(NPixel,NPixel),order='F')

    rewardimg = zeros((NPixel,NPixel))
    reward = zeros((PATCH_NUM))

    count=0
    for i in range(NPixel):
        for j in range(NPixel):
            temp = np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) - np.sum(dist2imgLarge[j:j + PATCH_reward, i:i + PATCH_reward])
            ########################## Reward 1 ############################
            # if np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])==0:
            #     if np.sum(dist2imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])==0:
            #         reward[count]=1
            #     else:
            #         reward[count]=-1
            #     count += 1
            # else:
            #     reward[count] = -np.sum(np.absolute(GTimgLarge[j:j + PATCH_reward, i:i + PATCH_reward])+1) / np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) + np.sum(np.absolute(GTimgLarge[j:j + PATCH_reward, i:i + PATCH_reward])+1) / np.sum(dist2imgLarge[j:j + PATCH_reward, i:i + PATCH_reward])
            #     count += 1
            #reward[count] = 1/(np.sum(dist2imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])+0.001) - 1/(np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])+0.001)
            #count = count+1


            ########################## Reward 2 ############################
            if np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])==0:
                if temp==0:
                    reward[count]=1
                else:
                    reward[count]=-1
                count += 1
            else:
                factor = 0.005
                if temp/np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])>=factor:
                    reward[count]=1
                if temp/np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])<factor and temp/np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])>=factor*0.1:
                    reward[count] = 0.5
                if temp/np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])<factor*0.1 and temp>0:
                    reward[count] = 0.1
                if temp==0:
                    reward[count] = 0
                if temp / np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) < 0:
                    reward[count] = -1
                count += 1
            rewardimg[i,j]= 1/(np.sum(dist2imgLarge[i:i+PATCH_reward,j:j+PATCH_reward])+0.001) - 1/(np.sum(dist1imgLarge[i:i+PATCH_reward,j:j+PATCH_reward])+0.001)
    reward = np.reshape(rewardimg,(PATCH_NUM),order='F')
    error = np.sum(np.absolute(dist2))


    return next_state, reward, para, gamma, error, fimg







def main():

    f = h5py.File('.../TrainData.mat', 'r')
    TrainData = f.get('/TrainData')
    TrainData = np.array(TrainData)

    f = h5py.File('.../TestData.mat', 'r')
    TestData = f.get('/TestData')
    TestData  = np.array(TestData)

    f = h5py.File('.../TrueImgTrain.mat', 'r')
    TrueImgTrain = f.get('/TrueImgTrain')
    TrueImgTrain  = np.array(TrueImgTrain)
    TrueImgTrain  = TrueImgTrain.transpose()

    f = h5py.File('.../TrueImgTest.mat', 'r')
    TrueImgTest = f.get('/TrueImgTest')
    TrueImgTest = np.array(TrueImgTest)
    TrueImgTest = TrueImgTest.transpose()

    f = h5py.File('.../pMat.mat', 'r')
    data = f['pMat']['data']
    ir = f['pMat']['ir']
    jc = f['pMat']['jc']
    pMat = scipy.sparse.csc_matrix((data, ir, jc))

    f = h5py.File('.../projdata_Train.mat','r')
    Projdata_Train = f.get('/projdata_Train')
    Projdata_Train = np.array(Projdata_Train)
    Projdata_Train = Projdata_Train.transpose()

    f = h5py.File('.../PTPN_Recon/projdata_Test.mat', 'r')
    Projdata_Test = f.get('/projdata_Test')
    Projdata_Test = np.array(Projdata_Test)
    Projdata_Test = Projdata_Test.transpose()

    save_session_name = 'Session/PTPN_Recon.ckpt'
    session_load_name = 'Session/PTPN_Recon.ckpt'
    start_time = time.time()

    with tf.Session() as sess:

        if load_session == 1:
            state_sel = np.load('.../replay_memory/state_PTPN_Recon.npy')
            next_state_sel = np.load('.../replay_memory/next_state_PTPN_Recon.npy')
            action_sel = np.load('.../replay_memory/action_PTPN_Recon.npy')
            reward_sel = np.load('.../replay_memory/reward_PTPN_Recon.npy')
            para_sel = np.load('.../replay_memory/para_PTPN_Recon.npy')
            count_memory = np.load('.../replay_memory/count_memory_PTPN_Recon.npy')
            indicator = np.load('.../replay_memory/indicator_PTPN_Recon.npy')
            load_episode = 19
        else:
            state_sel = np.zeros((REPLAY_MEMORY, INPUT_SIZE * INPUT_SIZE))
            next_state_sel = np.zeros((REPLAY_MEMORY, INPUT_SIZE * INPUT_SIZE))
            action_sel = np.zeros((REPLAY_MEMORY))
            reward_sel = np.zeros((REPLAY_MEMORY))
            done_sel = np.zeros((REPLAY_MEMORY))
            para_sel = np.zeros((REPLAY_MEMORY, INPUT_SIZE * INPUT_SIZE))
            count_memory = 0
            indicator = 0
            load_episode = 0

        mainDQN = dqn_cnn_iteration_till_end.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = dqn_cnn_iteration_till_end.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if load_session == 1:
            saver.restore(sess, session_load_name+str(load_episode+1))
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                     src_scope_name="main")
        sess.run(copy_ops)


        if MAX_EPISODES>0:
            reward_check = zeros(( MAX_EPISODES))
            Q_check =zeros(( MAX_EPISODES))
            Gamma = zeros((PATCH_NUM, 2, TRAIN_IMG_NUM))
            State = TrainData
            Para = 0.005 * ones((PATCH_NUM, TRAIN_IMG_NUM))
            itertotal = 50
            tol = zeros((TRAIN_IMG_NUM))
            tol[0] = 0.005
            tol[1] = 0.005
            tol[2] = 0.005
            tol[3] = 0.005
            tol[4] = 0.005
            tol[5] = 0.005
            for IMG in range(TRAIN_IMG_NUM):
                state = State[:, :, IMG]
                gamma = Gamma[:, :, IMG]
                GroundTruth = TrueImgTrain[:, IMG]
                projdata_Train = Projdata_Train[:, IMG]
                para = Para[:, IMG]
                action = 2 * ones((PATCH_NUM))
                next_state, reward, para, gamma, error, fimgIter = reconTV(pMat, projdata_Train, state, action,
                                                                           para, gamma, GroundTruth, NPixel,
                                                                           INPUT_SIZE, itertotal, tol[IMG])
                State[:, :, IMG] = next_state
                Gamma[:, :, IMG] = gamma

            State_initial = State

            for episode in range(MAX_EPISODES-load_episode-1):

                e = 0.999 / (((episode+load_episode) / 150) + 1)
                if e<0.1:
                    e=0.1
                step_count = 0
                State = State_initial
                Para = 0.005 * ones((PATCH_NUM, TRAIN_IMG_NUM))




                temp_reward=0
                temp_Q = 0
                for ITER_NUM in range(MAXITER_RECON):
                    for IMG_IDX in range(TRAIN_IMG_NUM):
                        state = State[:, :, IMG_IDX]
                        gamma = zeros((PATCH_NUM,2))
                        GroundTruth = TrueImgTrain[:,IMG_IDX]
                        projdata_Train  = Projdata_Train[:,IMG_IDX]
                        para = Para[:, IMG_IDX]
                        action = zeros((PATCH_NUM))
                        flag = np.random.rand(PATCH_NUM)
                        count_yy = 0
                        length_yy = 0
                        for idx in range(PATCH_NUM):
                            if flag[idx]>=e:
                                length_yy += 1
                        yy = zeros((length_yy, INPUT_SIZE*INPUT_SIZE))
                        for idx in range(PATCH_NUM):
                            if flag[idx]<e:
                                action[idx] = np.random.randint(OUTPUT_SIZE, size=1)
                            if flag[idx]>=e:
                                yy[count_yy, :] = state[idx,:]
                                count_yy += 1
                        action_yy = np.argmax(mainDQN.predict(yy),axis=3)
                        QvalueTemp = np.max(mainDQN.predict(yy),axis=3)
                        Qvalue = QvalueTemp[:,0,0]
                        action_yyy = action_yy[:,0,0]
                        avg_action = np.mean(action_yyy)
                        print('average action taken is: {}'.format(avg_action))
                        count_yy=0

                        for idx in range(PATCH_NUM):
                            if flag[idx] >= e:
                                action[idx] = action_yy[count_yy,0,0]
                                count_yy += 1
                        next_state, reward, para, gamma, error, fimgIter = reconTV(pMat,projdata_Train, state, action, para, gamma,GroundTruth,NPixel,INPUT_SIZE,itertotal,tol[IMG_IDX])

                        pl.figure('current results')
                        plt.subplot(131)
                        plt.imshow(log(np.reshape(para, (NPixel, NPixel), order='F')))
                        plt.subplot(132)
                        plt.imshow(
                            np.reshape(next_state[:, int((INPUT_SIZE * INPUT_SIZE + 1) / 2) - 1], (NPixel, NPixel),
                                       order='F'))
                        plt.subplot(133)
                        plt.imshow(np.reshape(GroundTruth, (NPixel, NPixel), order='F'))
                        plt.show(block=False)
                        plt.pause(0.1)

                        Para[:, IMG_IDX] = para

                        sel_prob = 0.01
                        flag1 = np.random.rand(PATCH_NUM)
                        flag2 = np.zeros([PATCH_NUM])
                        for idx in range(PATCH_NUM):
                            if flag1[idx]>=sel_prob:
                                flag2[idx] = 0
                            if flag1[idx]<sel_prob:
                                flag2[idx] = 1

                        sel_num = int(np.sum(flag2))

                        if count_memory+sel_num<=REPLAY_MEMORY-2:
                            for idx in range(PATCH_NUM):
                                if flag1[idx]<sel_prob:
                                    state_sel[count_memory,:] = state[idx,:]
                                    next_state_sel[count_memory,:] = next_state[idx,:]
                                    action_sel[count_memory]=action[idx]
                                    reward_sel[count_memory] = reward[idx]
                                    para_sel[count_memory] = para[idx]
                                    if ITER_NUM >= MAXITER_RECON-1:
                                        done_sel[count_memory] = 0
                                    if ITER_NUM < MAXITER_RECON - 1:
                                        done_sel[count_memory] = 1
                                    count_memory += 1
                        else:
                            indicator = 1
                            for idx in range(PATCH_NUM):
                                if flag1[idx]<sel_prob:
                                    state_sel[count_memory,:] = state[idx,:]
                                    next_state_sel[count_memory,:] = next_state[idx,:]
                                    action_sel[count_memory]=action[idx]
                                    reward_sel[count_memory] = reward[idx]
                                    para_sel[count_memory] = para[idx]
                                    if ITER_NUM >= MAXITER_RECON-1:
                                        done_sel[count_memory] = 0
                                    if ITER_NUM < MAXITER_RECON - 1:
                                        done_sel[count_memory] = 1
                                    if count_memory == REPLAY_MEMORY - 1:
                                        count_memory = 0
                                        print('Replay Memory is full')
                                    else:
                                        count_memory += 1
                        if indicator == 0:
                            replay_size = count_memory +  1
                        else:
                            replay_size = REPLAY_MEMORY

                        if replay_size > BATCH_SIZE:
                            if replay_size == REPLAY_MEMORY:
                                TRAIN_NUM_CURRENT = TRAIN_NUM_ITER*3
                            else:
                                TRAIN_NUM_CURRENT = TRAIN_NUM_ITER

                            for i in range(TRAIN_NUM_CURRENT):
                                shuffle_order = np.arange(replay_size)
                                np.random.shuffle(shuffle_order)
                                minibatch_state = state_sel[shuffle_order[0:BATCH_SIZE],:]
                                minibatch_next_state = next_state_sel[shuffle_order[0:BATCH_SIZE],:]
                                minibatch_action = action_sel[shuffle_order[0:BATCH_SIZE]]
                                minibatch_reward = reward_sel[shuffle_order[0:BATCH_SIZE]]
                                minibatch_para = para_sel[shuffle_order[0:BATCH_SIZE]]
                                minibatch_done = done_sel[shuffle_order[0:BATCH_SIZE]]

                                #minibatch = random.sample(replay_buffer, BATCH_SIZE)
                                loss, _ = replay_train(mainDQN, targetDQN, minibatch_state,minibatch_next_state,minibatch_action,minibatch_reward,minibatch_done, minibatch_para)
                                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                                    sess.run(copy_ops)
                                step_count += 1

                        State[:, :, IMG_IDX] = next_state

                    print("Episode: {}  Iterations: {} Loss: {}".format(episode, ITER_NUM, loss))

                CHECK = episode+1

                if save_session == 1 and CHECK % 20 == 0:
                    saver.save(sess, save_session_name, global_step=episode + 1)

                if save_session == 1 and CHECK % 20 ==0:
                    saver.save(sess, save_session_name, global_step=episode + 1)
                    np.save('.../replay_memory/state_PTPN_Recon', state_sel)
                    np.save('.../replay_memory/next_PTPN_Recon', next_state_sel)
                    np.save('.../replay_memory/action_PTPN_Recon', action_sel)
                    np.save('.../replay_memory/reward_PTPN_Recon', reward_sel)
                    np.save('.../replay_memory/para_PTPN_Recon', para_sel)
                    np.save('.../indicator_PTPN_Recon.npy',indicator)
                    np.save('.../replay_memory/count_memory_PTPN_Recon.npy',count_memory)

        print("--- %s seconds to do training ---" % (time.time() - start_time))

        # testing

        for IMG_IDX in range(TEST_IMG_NUM):
            tol = 0.001
            itertotal = 100
            state_test = TestData[:,:,IMG_IDX]
            projdata_Test = Projdata_Test[:,IMG_IDX]
            para_test = 1.5 * ones((PATCH_NUM))
            gamma = zeros((PATCH_NUM, 2))
            GroundTruth = TrueImgTest[IMG_IDX,:]
            action = 2*ones((PATCH_NUM))
            next_state_test, reward, para_test, gamma, error, fimg = reconTV(pMat, projdata_Test, state_test, action,
                                                                             para_test, gamma, GroundTruth, NPixel,
                                                                             INPUT_SIZE, itertotal, tol)
            state_test = next_state_test
            error_old = 1e5
            for ITER_NUM in range(MAXITER_RECON):
                X= state_test

                action1 = np.argmax(targetDQN.predict(X), axis=3)
                action2 = np.argmax(mainDQN.predict(X), axis=3)
                action = action2[:, 0, 0]
                print(np.mean(action))
                gamma = zeros((PATCH_NUM, 2))
                next_state_test, reward, para_test, gamma, error,fimg = reconTV(pMat, projdata_Test, X, action, para_test, gamma, GroundTruth, NPixel,INPUT_SIZE,itertotal,tol)
                pl.figure('current results')
                plt.subplot(121)
                plt.imshow(log(np.reshape(para_test, (NPixel, NPixel), order='F')))
                plt.subplot(122)
                plt.imshow(np.reshape(next_state_test[:,int((INPUT_SIZE*INPUT_SIZE+1)/2)-1], (NPixel, NPixel), order='F'))
                plt.show(block=False)
                plt.pause(0.2)

                print("Testing Image: {}, Iteration: {}, Mean testing error: {}".format(IMG_IDX, ITER_NUM, error))
                np.save('.../Test_results'+str(ITER_NUM),
                        state_test[:, int((INPUT_SIZE * INPUT_SIZE + 1) / 2) - 1])
                np.save('.../Test_para' + str(ITER_NUM),
                        para_test)
                state_test = next_state_test
                if error>error_old:
                    break
                error_old = error
            np.save('.../Test_results_'+str(IMG_IDX+1), fimg)
            np.save('.../Para_results_'+str(IMG_IDX+1), para_test)

if __name__ == "__main__":
    main()
