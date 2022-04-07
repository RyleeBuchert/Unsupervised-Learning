if __name__ == "__main__":

    input_means = "4.43469 13.40289 15.04757 9.53897 18.85027 1.43679 16.96103 12.78924 12.20637 6.44657"
    input_sd = "0.76385 0.17564 0.7548 1.34238 0.87502 1.04902 0.33685 0.16689 0.18383 0.44902"

    means_list = input_means.lstrip().rstrip().split(' ')
    means_list = [round(float(x), 3) for x in means_list]
    sorted_means = sorted(means_list)

    sd_list = input_sd.lstrip().rstrip().split(' ')
    sd_list = [round(float(x), 3) for x in sd_list]

    sorted_sd = []
    for val in sorted_means:
        sorted_sd.append(sd_list[means_list.index(val)])

    output_means = ""
    output_sd = ""
    for i in range(len(means_list)):
        if i == len(means_list) - 1:
            output_means += str(sorted_means[i])
            output_sd += str(sorted_sd[i])        
        else:
            output_means += str(sorted_means[i]) + ', '
            output_sd += str(sorted_sd[i]) + ', '

    print(output_means)
    print(output_sd)
