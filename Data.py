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
        for line in file:
            dis, cli, vec,p = processLine(line)
            display.append(dis)
            click.append(cli)
            user_vec.append(vec)
            pool.append(p)
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
    pool = np.array([int(i[3:-1]) for i in regions[2:]])
    return display,click,user_vec,pool

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

def process_large_data(data_dir,batch_size=200000,dump_folder='data'):
    '''Process a large data file (>200k lines) and dump them as npy files in batches
    
    @param
    data_dir -> [str] path to the data file for processing
    batch_size -> [int] number of entries one batch can contain. Default to 200k
    dump_folder -> [str] name of folder where dumps are saved

    '''
    batch_count=0
    item_count=0
    display = []
    click = []
    user_vec = []
    pool = []
    if os.path.isdir(dump_folder) is False:
        os.mkdir(dump_folder)
    with open(data_dir) as file:
        for line in file:
            dis, cli, vec,p = processLine(line)
            display.append(dis)
            click.append(cli)
            user_vec.append(vec)
            pool.append(p)
            item_count+=1
            if item_count >= batch_size:
                #dump current
                print("saving batch #{}".format(batch_count))
                display_path=os.path.join(dump_folder,'display'+str(batch_count))
                np.save(display_path,display)

                click_path=os.path.join(dump_folder,'click'+str(batch_count))
                np.save(click_path,click)

                user_path=os.path.join(dump_folder,'user'+str(batch_count))
                np.save(user_path,user_vec)

                pool_path=os.path.join(dump_folder,'pool'+str(batch_count))
                np.save(pool_path,pool)
                display = []
                click = []
                user_vec = []
                pool = []
                batch_count+=1
                item_count=0
    if display:
        print("saving batch #{}".format(batch_count))
        display_path=os.path.join(dump_folder,'display'+str(batch_count))
        np.save(display_path,display)

        click_path=os.path.join(dump_folder,'click'+str(batch_count))
        np.save(click_path,click)

        user_path=os.path.join(dump_folder,'user'+str(batch_count))
        np.save(user_path,user_vec)

        pool_path=os.path.join(dump_folder,'pool'+str(batch_count))
        np.save(pool_path,pool)
    return batch_count
def load_batch(n,dump_folder):
    display_path=os.path.join(dump_folder,'display'+str(n)+'.npy')
    click_path=os.path.join(dump_folder,'click'+str(n)+'.npy')
    user_path=os.path.join(dump_folder,'user'+str(n)+'.npy')
    pool_path=os.path.join(dump_folder,'pool'+str(n)+'.npy')
    display=np.load(display_path)
    click=np.load(click_path)
    user_vec=np.load(user_path)
    pool=np.load(pool_path)
    return display,click,user_vec,pool

def get_batched_data(batch_num,dump_folder='data'):
    '''A generator that load data from dump and yield one line of data at a time
    @param
    batch_num -> [int] total number of batches to read
    dump_foler -> [str] folder where the dumped files are. Default to "data"

    @yield
    yields a tuple with 4 elements (display,click,user_vec,pool), as numpy arrays

    Usage:
    data_gen=get_batched_data(n)
    for (display,click,user_vec,pool) in data_gen:
        #do something with current data
    '''
    assert batch_num>0
    if os.path.isdir(dump_folder) is False:
        raise ValueError("Invalide Dump folder")
    batch_count=0
    item_count=0
    #load batch 0
    display,click,user_vec,pool=load_batch(0,dump_folder)
    
    while batch_count<batch_num:
        yield display[item_count],click[item_count],user_vec[item_count],pool[item_count]
        item_count+=1
        if item_count>=len(display):
            #get next batch
            batch_count+=1
            item_count=0
            display,click,user_vec,pool=load_batch(0,dump_folder)
        


if __name__=='__main__':

    data_dir ="data.txt" #"ydata-fp-td-clicks-v2_0.20111003"
    #rewrite(data_dir,200000)
    n=process_large_data(data_dir,batch_size=100,dump_folder='batch')
    print(n)
    data_gen=get_batched_data(n,dump_folder='batch')
    count=0
    for (display,click,user_vec,pool) in data_gen:
        count+=1
        if click>0:
            print(click)
    print(count)