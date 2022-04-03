import pandas as pd


class Gaussian_Data():

    def __init__(self):
        self.gaussian_type = {}
        self.results_dict = {}

    def read_gaussian(self, num):
        with open(f'Data\\Gaussian_Sources\\File{num}.txt', 'r') as file:
            line = file.readline()
            line_type = line[0:8]
            gen_count = 0
            while(line_type == "Gaussian"):
                data_info = line.split(" ")
                data_stats = data_info[2].split(',')

                mean_flag = False
                mean_string = ""
                for char in data_stats[0]:
                    if mean_flag:
                        mean_string += char
                    if char == '=':
                        mean_flag = True

                sd_flag = False
                sd_string = ""
                data_stats[1] = data_stats[1].replace('\n', '')
                for char in data_stats[1]:
                    if sd_flag:
                        sd_string += char
                    if char == '=':
                        sd_flag = True

                gen_count += 1
                self.gaussian_type.update({gen_count: {'mean': float(mean_string), 'sd': float(sd_string)}})
                
                line = file.readline()
                line_type = line[0:8]

            data = []
            while(line != ''):
                line = line.replace('\n', '')
                data.append([float(i) for i in line.split('\t')])
                line = file.readline()
            cols = ['mean', 'point']
            return [gen_count, self.gaussian_type, pd.DataFrame(data, columns=cols)]