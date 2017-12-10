import numpy as np
import os
import os.path
USER_VEC_SIZE=136

def load_data(data_dir):
    '''
    data_dir: director path of data
    return->Tuple(display,click,user_vec,pool):
        display[str]: the article ID present
        click[int]: if the presented article is clicked, return 0, otherwise, return 1
        user_vec[1D-array]: user feature vector
        pool[str list]: list of article ID
    '''
    display = []
    click = []
    user_vec = []
    pool = []
    with open(data_dir) as file:
        first_line = file.readline()
        regions=first_line.split('|')
        pool = [int(i[3:-1]) for i in regions[2:]]
    with open(data_dir) as file:
        for line in file:
            
            dis, cli, vec = processLine(line)
            try:
                display.append(pool.index(dis))
            except ValueError:
                continue
            click.append(cli)
            user_vec.append(vec) 
    return display,click,user_vec,pool

def get_pool(file):
    '''
    file: an read-only opened file
    pool[str list]: list of article ID
    '''
    first_line = file.readline()
    regions=first_line.split('|')
    pool = [int(i[3:-1]) for i in regions[2:]]
    return pool

def process_line(file_line,pool):
    '''
    pool[str list]: list of article ID
    data_dir: director path of data
    return->Tuple(display,click,user_vec,pool):
        dis[str]: the article ID present
        cli[int]: if the presented article is clicked, return 0, otherwise, return 1
        vec[1D-array]: user feature vector
    '''    
    dis, cli, vec = processLine(file_line)
    dis = pool.index(dis)
    return dis, cli, vec

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

def rewrite(data_dir,i):
    '''
    get in a large data file and create a file named rewrite.txt
    i: the number of lines from source data file
    '''
    wfile = open("rewrite.txt","w")
    l = 0
    with open(data_dir) as file:
        for line in file:
            wfile.write(line)
            if(l<i):
                l+=1
            else:
                wfile.close()
                return
    
def dump_to_file(display,click,user_vec,pool):
    if os.path.isdir('tmp') == False:
        os.mkdir('tmp')
    display_path=os.path.join('tmp','display')
    np.save(display_path,display)

    click_path=os.path.join('tmp','click')
    np.save(click_path,click)

    user_path=os.path.join('tmp','user')
    np.save(user_path,user_vec)

    pool_path=os.path.join('tmp','pool')
    np.save(pool_path,pool)

def load_from_dump(base_dir='tmp'):
    if os.path.isdir(base_dir)==False:
        raise ValueError("Invalid dump directory")
    display_path=os.path.join(base_dir,'display.npy')
    click_path=os.path.join(base_dir,"click.npy")
    user_path=os.path.join(base_dir,'user.npy')
    pool_path=os.path.join(base_dir,'pool.npy')

    display=np.load(display_path)
    click=np.load(click_path)
    user_vec=np.load(user_path)
    pool=np.load(pool_path)
    return display,click,user_vec,pool
if __name__=='__main__':

    data_dir = "ydata-fp-td-clicks-v2_0.20111003"
    rewrite(data_dir,200000)
    display,click,user_vec,pool = load_data("rewrite.txt")
    dump_to_file(display,click,user_vec,pool)

    display,click,user_vec,pool=load_from_dump()
    