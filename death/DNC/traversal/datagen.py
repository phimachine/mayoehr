import numpy as np
from multiprocessing import Pool, Lock
from threading import Thread
from numpy.random import uniform, choice
from sklearn.neighbors import NearestNeighbors
import archi.param as param
import time

# class Graph:
#     def __init__(self):
#         self.nodes=[]
#
# class Node:
#     def __init__(self):
#         self.label=None
#         self.edges=[]
#
#
#
# class Edge:
#     def __init__(self,label,target):
#         self.label=None
#         self.target=None
lock=Lock()

class Datagen():
    def __init__(self,diff=1):
        # the traversal datagen follows curriculum learning outlined in the paper.

        # linear lesson difficulty
        self.diff=diff
        self.node_count=None
        self.outdeg=None
        self.travlen=None
        self.update_param()

    def update_param(self):
        # number of the nodes in the graph
        self.node_count=16*self.diff

        # k nearest neighbour
        # outbound degree
        # 1/8 of the total nodes
        self.outdeg=2*self.diff

        # traversal length
        # 1/4 of the total nodes
        self.travlen=4*self.diff

    def change_diff(self,new_diff):
        self.diff=new_diff
        self.update_param()

    def path_gen(self):
        # sample uniformly from a 2-d square

        x=uniform(0,1,self.node_count)
        y=uniform(0,1,self.node_count)
        graph=np.array([x,y]).transpose()
        nbrs = NearestNeighbors(n_neighbors=self.outdeg+1, algorithm='ball_tree').fit(graph)
        distances, indices = nbrs.kneighbors(graph)
        # apparently the nearest neighbour is the point itself
        indices=indices[:,1:]

        # replace indices to labels
        newneighbours=[]
        for source in indices:
            newsource=[]
            # two source nodes can have exactly the same edge labels, as outlined in the paper
            edge_labels=choice(range(1000), size=self.outdeg, replace=False)
            for i,outbound in enumerate(source):
                node_edge=[outbound, edge_labels[i]]
                newsource.append(node_edge)
            newneighbours.append(newsource)
        newneighbours=np.array(newneighbours)

        path=[]
        prev_node=choice(self.node_count,size=1)[0]
        for _ in range(self.travlen):
            node_edge=newneighbours[prev_node,choice(range(self.outdeg),size=1)].squeeze(0)
            triple=[prev_node,node_edge[1],node_edge[0]]
            prev_node=node_edge[0]
            path.append(triple)

        # translate to triples
        point_labels = choice(range(1000), size=self.node_count, replace=False)
        for triple in path:
            triple[0]=point_labels[triple[0]]
            triple[2]=point_labels[triple[2]]

        onehot_path=[]
        for triple in path:
            # for input
            vec = np.zeros(92, dtype=np.float32)
            for i,num in enumerate(triple):
                vec[int(num//100+i*30)]=1.0
                vec[int(num//10%10+10+i*30)]=1.0
                vec[int(num%10+20+i*30)]=1.0
            onehot_path.append(vec)

        # indicate the start of the answer phase
        onehot_path[-1][90]=1
        onehot_path=np.array(onehot_path)
        answer_phase_input=np.zeros((self.travlen,92),dtype=np.float32)
        answer_phase_input[:,91]=1.0

        # target_output
        # with mask, only the answer phase will count
        description_phase_target=np.zeros((self.travlen,9))
        answer_phase_target=onehot_path[:,:90].copy()
        answer_phase_target=answer_phase_target.reshape(self.travlen,9,10)
        answer_phase_target=np.argmax(answer_phase_target,axis=2)
        target=np.concatenate((description_phase_target,answer_phase_target))

        onehot_path[:,60:90]=0
        input=np.concatenate((onehot_path,answer_phase_input))

        critical_index=np.arange(self.travlen,self.travlen*2,1)

        return input,target,critical_index

    def pathgen_helper(self,_):
        return self.path_gen()

    def datagen(self, batch_size):

        batch_input=np.zeros((batch_size,self.travlen*2,92))
        # print(batch_input.shape)
        batch_target=np.zeros((batch_size,self.travlen*2,9))
        batch_critical=np.zeros((batch_size,self.travlen))
        for i in range(batch_size):
            input, target, critical_index= self.path_gen()
            batch_input[i,:,:]=input
            batch_target[i,:,:]=target
            batch_critical[i,:]=critical_index

        return batch_input,batch_target,batch_critical

        # below code does not work. Apparently when datagen() is calle twice in
        # a short period of time, batch_input is shared. What the heck?

        # with Pool() as pool:
        #     result=pool.map(self.pathgen_helper, [None]*batch_size)
        #     for i,val in enumerate(result):
        #         try:
        #             batch_input[i,:,:]=val[0]
        #         except ValueError:
        #             print("damn")
        #         batch_target[i,:,:]=val[1]
        #         batch_critical[i,:]=val[2]
        # return batch_input,batch_target,batch_critical

class PreGenData(Datagen):
    # the purpose of this class is to generate data before it's required to use.
    # this will reduce 11% of my code run time according to cProfiler.
    def __init__(self, batch_size):
        super(PreGenData, self).__init__(1)
        self.batch_size = batch_size
        self.val_ready = False
        self.train_ready = False
        self.next_train = None
        self.next_validate = None
        self.__gendata_train()
        self.__gendata_val()


    def get_train(self):
        Thread(target=self.__gendata_train).start()
        while not self.train_ready:
            print('train data is not ready?')
            time.sleep(1)
        return self.next_train

    def get_validate(self):
        Thread(target=self.__gendata_val).start()
        while not self.val_ready:
            print('val data is not ready?')
            time.sleep(1)
        return self.next_train

    def __gendata_train(self):
        self.next_train = self.datagen(self.batch_size)
        self.train_ready = True

    def __gendata_val(self):
        self.next_validate = self.datagen(self.batch_size)
        self.val_ready = True

    def change_diff(self,new_diff):
        super(PreGenData, self).change_diff(new_diff)
        self.val_ready = False
        self.train_ready = False
        self.next_train = None
        self.next_validate = None
        self.__gendata_train()
        self.__gendata_val()


if __name__=="__main__":
    # pgd=PreGenData(3)
    # pgd.path_gen()

    pgd=PreGenData(3)
    print("default diff:\n",pgd.get_train())
    pgd.change_diff(2)
    print("diff 2:\n", pgd.get_train())