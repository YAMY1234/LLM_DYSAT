import numpy as np

def loadratings():
    file_path = 'u.data'
    dat_set = np.loadtxt(file_path, dtype=int, delimiter='\t', skiprows=1, usecols=(0, 1, 3), unpack=False)
    dat_set[:, :2] -= 1

    data_user = dat_set[:, 0]
    dat_set_list = list(dat_set)
    dat_set_time = sorted(dat_set_list, key=lambda x: x[2])

    idx = np.array([i for i in range(len(data_user))])
    label = np.array([1 for i in range(len(data_user))])

    dat_set = np.hstack([np.reshape(dat_set_time, [-1, 3]), np.reshape(label, [-1, 1]), np.reshape(idx, [-1, 1])])

    data_train = dat_set[:int(dat_set.shape[0] * 0.8), :]
    data_val = dat_set[int(dat_set.shape[0] * 0.8):int(dat_set.shape[0] * 0.9), :]
    data_test = dat_set[int(dat_set.shape[0] * 0.9):, :]

    np.savetxt('train.txt', data_train, fmt='%d', delimiter=',', encoding='utf-8')
    np.savetxt('val.txt', data_val, fmt='%d', delimiter=',', encoding='utf-8')
    np.savetxt('test.txt', data_test, fmt='%d', delimiter=',', encoding='utf-8')
    np.savetxt('ml-100k.txt', dat_set, fmt='%d', delimiter=',', encoding='utf-8')
def u_attribute():
    file_path = 'u.user'
    dat_set = np.loadtxt(file_path, dtype=str, delimiter='|', skiprows=0, usecols=(0, 1, 2, 3), unpack=False)
    dat_user = dat_set[:, 0].astype(int)-1
    # dat_set[:, 0] -= 1

    dat_age = dat_set[:, 1]
    dat_xingbe = dat_set[:, 2]
    dat_occupation = dat_set[:, 3]

    age_list = []
    for age in dat_age.astype(int):
        if age<10:
            age_list.append(0)
        elif age<20:
            age_list.append(1)
        elif age<30:
            age_list.append(2)
        elif age<40:
            age_list.append(3)
        elif age<50:
            age_list.append(4)
        elif age<60:
            age_list.append(5)
        elif age<70:
            age_list.append(6)
        elif age<80:
            age_list.append(7)


    age_map = {j: i for i, j in enumerate(sorted(set(dat_age.astype(int))))}  # 对菜品进行去重编号
    xingbie_map = {j: i for i, j in enumerate(set(dat_xingbe))}  # 对菜品进行去重编号
    occupation_map = {j: i for i, j in enumerate(set(dat_occupation))}  # 对菜品进行去重编号

    # idx_age = np.array(list(map(age_map.get, dat_age)))
    idx_age = np.array(age_list)
    idx_xingbie = np.array(list(map(xingbie_map.get, dat_xingbe)))
    idx_occupation = np.array(list(map(occupation_map.get, dat_occupation)))

    dat_set_attributes = np.dstack((dat_set[:, 0].astype(int), idx_age, idx_xingbie, idx_occupation))[0]  # 横向拼接  生成商家-菜品对

    np.savetxt('user_attribute.txt', dat_set_attributes, fmt='%d', delimiter=',', encoding='utf-8')



if __name__ == '__main__':
    # loadusers()
    # loadmovies()
    u_attribute()