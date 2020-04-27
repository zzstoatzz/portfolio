import os, sys, copy, random, time
WIDTH, HEIGHT = 128, 1 #128->256
W, N = (WIDTH - 1), WIDTH*WIDTH*HEIGHT*3-WIDTH*WIDTH
PRINTFREQ, MAX = 10, 100
# 'union' subroutine to connect vertices
def connect(v1, v2, A, B, i):
    v1[i] = A
    v2[i] = B        
    return i + 1
# recursive path compression 'find' algorithm
def findroot(i, ptr):
    if ptr[i] < 0: 
        return i
    ptr[i] = findroot(ptr[i], ptr)
    return ptr[i]
# init vertex lists, return iterator (3/2*N) as 'index'
def init_lists(v1, v2):
    index = 0
    for x in range(0, WIDTH):
        for y in range(0, WIDTH):
            for z in range(0, HEIGHT):
                s1 = x + WIDTH*y + WIDTH*WIDTH*z
                s2 = ((x+1)&W) + WIDTH*y + WIDTH*WIDTH*z
                index = connect(v1, v2, s1, s2, index)
                s2 = x + WIDTH*((y+1)&W)+ WIDTH*WIDTH*z
                index = connect(v1, v2, s1, s2, index)
                if (z < HEIGHT-1):
                    s2 = x + WIDTH*y + WIDTH*WIDTH*((z+1)&W)
                    index = connect(v1, v2, s1, s2, index)
                else:
                    s2 = x + WIDTH*y + WIDTH*WIDTH*(0)
                    index = connect(v1, v2, s1, s2, index)
    return index
# randomize list of bonds
def shuffle_bonds(index, v1, v2):
    for i in range(0, index):
        rn = random.random()
        j = i + int((index-i)*rn)
        temp = copy.copy(v1[i])
        v1[i] = v1[j]
        v1[j] = temp
        temp = copy.copy(v2[i])
        v2[i] = v2[j]
        v2[j] = temp
# union-find routine: bonds connect sites
def cluster(index, lists, big, M2):
    v1, v2, ptr, smax, M2tot, M2minus = lists
    for i in range(0, index):
        r1 = findroot(v1[i], ptr)
        r2 = findroot(v2[i], ptr)
        if (r2 != r1):
            M2 += ptr[r1]*2.0*ptr[r2]
            if ptr[r1] > ptr[r2]:
                ptr[r2] += ptr[r1]
                ptr[r1] = r2
                r1 = r2
            else:
                ptr[r1] += ptr[r2]
                ptr[r2] = r1
            if (-ptr[r1]>big):
                big = -ptr[r1]
        smax[i] += big
        M2tot[i] += M2
        M2minus[i] += (M2 - big*1.0*big)
    return ptr
# print to outfile
def fprintf(out, qs, run, index):
    smax, M2tot, M2minus = qs
    if (run % PRINTFREQ == 0):
        f = open(out, "w")
        for i in range(0, index):
            s = round(smax[i]/run, 4)
            m = round(M2tot[i]/run, 4)
            m_ = round(M2minus[i]/run, 4)
            line = str(s) + " " + str(m) + " " + str(m_) + "\n" 
            print(line, file=f)
        f.close()

def main():
    out = "outfiles/bond_mod"
    v1, v2  = [0]*(3*N), [0]*(3*N)
    index = init_lists(v1, v2)                             #--see above--#
                           
    smax, M2tot, M2minus = [[0]*index for i in range(0, 3)]#init Q lists
    print("Num bonds:", N)                                 

    for run in range(1, MAX):                              # main iterator for all runs
        if run % PRINTFREQ == 0: print("runs ", run)       # update # runs in terminal

        shuffle_bonds(index, v1, v2)                       #--see above--#

        M2, ptr, big = N, [-1]*N, 0                        # init M2, ptr lists and biggest cluster 
        lists = [v1, v2, ptr, smax, M2tot, M2minus]        # pack lists for cleaner f(args)
        ptr = cluster(index, lists, big, M2)                     #--see above--#

        qs = lists[3:]                                     # slice list of lists for fprintf
        fprintf(out, qs, run, index)                       #--see above--#
begin = time.time()
main()
exe_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-begin))
print("Execution time for width, height =",(WIDTH, HEIGHT), "is:" , exe_time)