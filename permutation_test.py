import pandas as pd
import numpy as np
import treeswift # please cite https://github.com/niemasd/TreeSwift 
from queue import PriorityQueue,Queue
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse 
import os
import sys

parser = argparse.ArgumentParser(
                    prog='Clustering significance test',
                    description='Permutation test to test the statistical significance of clustering of different metadata annotations.',
                    epilog='https://github.com/Yasas1994/Clustering-significance-test')

parser.add_argument('-t','--tree', help='tree file in nexus format', required=True)  
parser.add_argument('-m', '--meta' , help='metadata file in tsv file', required=True)    
parser.add_argument('-i1', help='index of the column with tree lables' ,type=int) 
parser.add_argument('-i2', help='index of the column with annotation lables to test', type=int) 
parser.add_argument('-p','--p_value',help='significance level for cluster selection', default=0.05, type=float)
parser.add_argument('-r','--replicates',help='number of permutation replicates', default=10000, type=int)
parser.add_argument('-w','--workers',help='number of workers', default=10, type=int)
parser.add_argument('-o','--out', help='output directory path', required=True)
args = parser.parse_args()

if not os.path.exists(args.out):
   os.makedirs(args.out)

# these functions were grabbed from https://github.com/niemasd/TreeCluster/blob/master/TreeCluster.py
# please cite this paper if you use this script 
def root_dist(tree,threshold,support):
    leaves = prep(tree,support)
    clusters = list()
    for node in tree.traverse_preorder():
        # if I've already been handled, ignore me
        if node.DELETED:
            continue
        if node.is_root():
            node.root_dist = 0
        else:
            node.root_dist = node.parent.root_dist + node.edge_length
        if node.root_dist > threshold:
            cluster = cut(node)
            if len(cluster) != 0:
                clusters.append(cluster)
                for leaf in cluster:
                    leaves.remove(leaf)

    # add all remaining leaves to a single cluster
    if len(leaves) != 0:
        clusters.append(list(leaves))
    return clusters
# initialize properties of input tree and return set containing taxa of leaves
def prep(tree, support, resolve_polytomies=True, suppress_unifurcations=True):
    if resolve_polytomies:
        tree.resolve_polytomies()
    if suppress_unifurcations:
        tree.suppress_unifurcations()
    leaves = set()
    for node in tree.traverse_postorder():
        if node.edge_length is None:
            node.edge_length = 0
        node.DELETED = False
        if node.is_leaf():
            leaves.add(str(node))
        else:
            try:
                node.confidence = float(str(node))
            except:
                node.confidence = 100. # give edges without support values support 100
            if node.confidence < support: # don't allow low-support edges
                node.edge_length = float('inf')
    return leaves
def cut(node):
    cluster = list()
    descendants = Queue(); descendants.put(node)
    while not descendants.empty():
        descendant = descendants.get()
        if descendant.DELETED:
            continue
        descendant.DELETED = True
        descendant.left_dist = 0; descendant.right_dist = 0; descendant.edge_length = 0
        if descendant.is_leaf():
            cluster.append(str(descendant))
        else:
            for c in descendant.children:
                descendants.put(c)
    return cluster

tree = treeswift.read_tree_nexus(args.tree)
annot = pd.read_table(args.meta, header=None)

annot.rename(columns={args.i1 : 'leaf_lab'}, inplace=True)
#print(annot.head())
#annot.columns = ['seqid','annot_1','annot_2','leaf_lab']
#clust = root_dist(tree['tree_1'], 4, 1 ) #get distance based clusters


#get all internal nodes from a tree. consider each branch as a cluster
clust_n = 0
clust_leaf = []
for branch in tree['tree_1'].traverse_postorder():
       if not branch.is_leaf():
            tmp_ = []
            for leaf in branch.traverse_leaves():
                tmp_.append([clust_n,str(leaf)])
            if len(tmp_) > 10: #extract all branches with more than n leaves
                #print(len(tmp_))
                clust_leaf.extend(tmp_)
                clust_n += 1


clust_leaf=pd.DataFrame(clust_leaf)
#print(clust_leaf.head())
clust_leaf.columns = ['cluster', 'leaf_lab']
#print(clust_leaf.head())
annot = pd.merge(clust_leaf,annot,
                 right_on='leaf_lab', 
                 left_on='leaf_lab') #add annotations to clusters

column = args.i2 #only chnage this colum 
#column2 = 'tmp' #simulation output is saved here

tmp = annot.groupby('cluster')[column].value_counts().sort_index()
tmp = pd.DataFrame(tmp)
tmp.columns = ['counts']
tmp.reset_index(inplace=True)

sum_cluster = tmp.groupby('cluster').sum(numeric_only=True)
max_per_cluster = tmp.groupby('cluster').max()
observed_difference_in_nps=sum(max_per_cluster['counts'])/sum(sum_cluster['counts']) #cluster purity https://stats.stackexchange.com/questions/95731/how-to-calculate-purity
observed_difference_per_cluster = np.array(max_per_cluster['counts']/sum_cluster['counts'])

print(f"global cluster purity: {observed_difference_in_nps : .2f}", file=sys.stderr)

#simulation
simulated = []
simulated_per_group = []
# for _ in tqdm(range(args.replicates)):
def simulate(annot):
    annot = annot.copy()
    annot['tmp'] = annot[column].sample(frac=1).values
    tmp2 = annot.groupby('cluster')['tmp'].value_counts().sort_index()
    tmp2 = pd.DataFrame(tmp2)
    tmp2.columns = ['counts']
    tmp2.reset_index(inplace=True)
    sum_cluster2 = tmp2.groupby('cluster').sum(numeric_only=True)
    max_per_cluster2 = tmp2.groupby('cluster').max()
    # simulated_per_group.append(max_per_cluster2['counts']/sum_cluster2['counts'])
    # simulated.append(sum(max_per_cluster2['counts'])/sum(sum_cluster2['counts']))
    return [max_per_cluster2['counts']/sum_cluster2['counts'],sum(max_per_cluster2['counts'])/sum(sum_cluster2['counts'])]

with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit tasks to the executor
    futures = [executor.submit(simulate,annot) for _ in range(args.replicates)]
    
    # Retrieve and print the results
    for future in tqdm(as_completed(futures),total=args.replicates):
        result = future.result()
        simulated_per_group.append(result[0])
        simulated.append(result[1])

simulated_results_per_cluster = np.array(simulated_per_group)
print(f"average global cluster purity (permuted) : { np.mean(simulated):.2f} {u'±'}{np.std(simulated) : .2f}",file=sys.stderr)

significance_level = args.p_value

simulations_greater_than_observed_cluster= sum(
    simulated_results_per_cluster >= observed_difference_per_cluster
)
num_simulations_cluster = simulated_results_per_cluster.shape[0]
p_value = simulations_greater_than_observed_cluster / num_simulations_cluster
# Boolean which is True if significant, False otherwise
significant_or_not_cluster = p_value < significance_level

per_clust = pd.DataFrame(zip(np.arange(significant_or_not_cluster.shape[0]),significant_or_not_cluster,p_value, annot['cluster'].value_counts().sort_index().to_list()))
per_clust.columns = ['clusters','is_significant','p_value','cluster_size']

per_clust.query('is_significant == True').to_csv(os.path.join(args.out,'significant_clusters.csv'),index=None)
annot.to_csv(os.path.join(args.out,'clusters.csv'),index=None)

#Global cluster purity (Permuted)
simulated_results=np.array(simulated)
significance_level = args.p_value

# check whether the cluster purity increases with shuffling - ideally the cluster purity should decrease 
simulations_greater_than_observed= sum(
    simulated_results >= observed_difference_in_nps
)
num_simulations = simulated_results.shape[0]
p_value = simulations_greater_than_observed / num_simulations
significant_or_not = p_value < significance_level

# Plot permutation simulations
density_plot = sns.kdeplot(simulated, fill=True, label='Permuted')
density_plot.set(
    xlabel='Absolute Difference cluster purity',
    ylabel='Proportion of Simulations',
    title=f'Permutation test for determination\n of cluster-label congruence \n{ "Test: Passed" if significant_or_not else "Test:Failed"} (p = {p_value})'
    
)

# Add a line to show the actual difference observed in the data
density_plot.axvline(
    
    x=observed_difference_in_nps, 
    color='red', 
    linestyle='--',
    label = 'Observed Difference'
)

plt.legend(
    loc='upper right'
)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(args.out,'global_significance.png'))