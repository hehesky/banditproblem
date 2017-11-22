import numpy as np

USER_VEC_SIZE=136
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
    
    pool_size=len(regions[2:])+1
    return display,click,user_vec,pool_size


if __name__=='__main__':
    line = '1318722897 id-611110 0 |user 1 6 11 13 33 18 14 39 20 |id-596821 |id-600025 |id-604964 |id-605378 |id-605423 |id-605518 |id-605672 |id-606079 |id-606207 |id-606696 |id-606812 |id-606900 |id-610233 |id-610351 |id-610505 |id-610758 |id-611078 |id-611110 |id-611479 |id-611482 |id-611585 |id-611775 |id-611932 |id-612378 |id-612461 |id-612506 |id-613111 |id-613241 |id-613242 |id-613404 |id-613449 |id-613505 |id-613546 |id-613675 |id-613689 |id-613765 |id-613786'
    display,click,user_vec,pool_size=processLine(line)
    print display
    print click
    print user_vec
    print pool_size

    line='1318722895 id-612378 0 |user 1 10 12 13 16 18 15 14 |id-596821 |id-600025 |id-604964 |id-605378 |id-605423 |id-605518 |id-605672 |id-606079 |id-606207 |id-606696 |id-606812 |id-606900 |id-610233 |id-610351 |id-610505 |id-610758 |id-611078 |id-611110 |id-611479 |id-611482 |id-611585 |id-611775 |id-611932 |id-612378 |id-612461 |id-612506 |id-613111 |id-613241 |id-613242 |id-613404 |id-613449 |id-613505 |id-613546 |id-613675 |id-613689 |id-613765 |id-613786'
    display,click,user_vec,pool_size=processLine(line)
    print display
    print click
    print user_vec
    print pool_size
