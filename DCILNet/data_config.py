
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = 'D:\LJH\SHUJUJI\LEVIR8'

        elif data_name == 'DECD':
            self.label_transform = "norm"
            self.root_dir = 'D:/LJH/SHUJUJI/DECD_cut256'
        elif data_name == 'GG':
            self.root_dir = 'D:/LJH/myproject/HZW2/'
        else:

            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='DSIFN')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

