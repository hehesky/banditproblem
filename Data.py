import numpy as np

USER_VEC_SIZE=136

def load_data(data_dir):
    '''
    Loads all data from the given data directory.
    and return lines from data file
    '''
    display = []
    click = []
    user_vec = []
    pool = []
    with open(data_dir) as file:
        for line in file:
            dis, cli, vec = processLine(line)
            display.append(dis)
            click.append(cli)
            user_vec.append(vec) 
    with open(data_dir) as file:
        first_line = file.readline()
        regions=first_line.split('|')
        pool = [int(i[3:-1]) for i in regions[2:]]
        
    return first_line,display,click,user_vec,pool

def processLine(line):
    regions=line.split('|')
    seg=regions[0].split(' ')
    display=seg[1]
    click=seg[2]
    display=int(display[3:])
    click=int(click)

    user_raw=regions[1]

    user=user_raw.split(' ')
    user=np.array(user[1:-1]).astype('int')
    user_vec=np.zeros(USER_VEC_SIZE)
    for i in user:
        user_vec[i-1]=1
        
    return display,click,user_vec

def rewrite(data_dir):
    wfile = open("rewrite.txt","w")
    l = 0
    with open(data_dir) as file:
        for line in file:
            wfile.write(line)
            if(l<1000):
                l+=1
            else:
                wfile.close()
                return
    return

if __name__=='__main__':

    line = '1318722897 id-611110 0 |user 1 6 11 13 33 18 14 39 20 |id-596821 |id-600025 |id-604964 |id-605378 |id-605423 |id-605518 |id-605672 |id-606079 |id-606207 |id-606696 |id-606812 |id-606900 |id-610233 |id-610351 |id-610505 |id-610758 |id-611078 |id-611110 |id-611479 |id-611482 |id-611585 |id-611775 |id-611932 |id-612378 |id-612461 |id-612506 |id-613111 |id-613241 |id-613242 |id-613404 |id-613449 |id-613505 |id-613546 |id-613675 |id-613689 |id-613765 |id-613786'
    display,click,user_vec=processLine(line)
    print(display)
    print(click)
    print(user_vec)

    data_dir = "data.txt"
    #rewrite(data_dir)
    first_line,display,click,user_vec,pool = load_data(data_dir)
    