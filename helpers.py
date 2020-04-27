import pandas as pd
import matplotlib.pyplot as plt
from findiff import FinDiff as fd 

def display(outfile):
    df = pd.read_csv(outfile,sep=' ', header=None)
    df.columns = ['smax', 'M2', 'M2-']
    plt.plot(df.index.values/df.shape[0], df['smax'])
    plt.title('Bond Percolation: Maximum Cluster Size')
    plt.xlabel('Occupation Probability p')
    plt.ylabel('# Nodes in Largest Cluster')
    # print threshold probability value
    M = df['smax']
    d_di = fd(0, 1)
    dm_di = d_di(M, 1)
    cp = dm_di.tolist().index(max(dm_di))
    print("\n", "p_c_i =",round(cp/df.shape[0], 5))

def init_g(WIDTH, HEIGHT):
    g = i.Graph()
    g.add_vertices(WIDTH*WIDTH*HEIGHT)
    style = {}
    #style["vertex_size"] = 10
    style["layout"] = 'grid_3d'
    #style["margin"] = 50
    #style["autocurve"] = False
    return g, style
  
def plot(g, style, j):
    return i.plot(g, **style).save("imgs/img"+str(j+1)+".png")
    
def valid_pair(s, t, WIDTH):
    distinct = s != t
    x_per = abs(s-t) != WIDTH-1
    y_per = abs(s-t) != (WIDTH-1)*WIDTH
    return distinct and x_per and y_per

def get_colors(vs, colors, ptr):
    b_g, g_r = colors
    c = []
    for i in vs:
        if ptr[i] < 0:
            c.append(str(b_g[len(b_g)+ptr[i]]))
        else:
            c.append(str(g_r[ptr[i]]))
    return c