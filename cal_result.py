from openpyxl import load_workbook

sum_of_detection = 0
sum_of_tracking = 0

sum_of_under_detection = 0
sum_of_warning = 0
sum_of_coming = 0
all_row = 0
for j in range(1, 11):
    # sum_of_detection = 0
    # sum_of_tracking = 0
    # sum_of_under_detection = 0
    # sum_of_warning = 0
    # sum_of_coming = 0

    whole_frame_count = 0
    file_name = './result/result' + str(j) + '.xlsx'
    load_wb = load_workbook(file_name, data_only=True)

    load_wb = load_wb['Sheet']
    cell_all_detection = load_wb['A']
    cell_all_tracker = load_wb['B']
    cell_come_num = load_wb['C']
    cell_warn_num = load_wb['D']
    cell_under_detected = load_wb['E']

    for i in range(1, len(cell_all_detection)):
        all_row += 1
        if cell_all_detection[i].value == cell_under_detected[i].value and cell_come_num[i].value + cell_warn_num[
            i].value == 0:
            continue
        if cell_all_detection[i].value == 0:
            continue  # frame detection x, under detection x

        whole_frame_count += 1
        sum_of_detection += cell_all_detection[i].value
        sum_of_tracking += cell_all_tracker[i].value
        # print(cell_come_num[i].value)
        # print(cell_warn_num[i].value)
        # print(cell_under_detected[i].value)
        if cell_all_detection[i].value != 0 and cell_under_detected[i] != 0:
            sum_of_under_detection += cell_under_detected[i].value
            sum_of_coming += cell_come_num[i].value
            sum_of_warning += cell_warn_num[i].value

print("all : ", all_row)
print('acc_1', sum_of_tracking / sum_of_detection)
print('acc_2', (sum_of_coming + sum_of_warning) / sum_of_under_detection)
error = sum_of_under_detection - (sum_of_warning + sum_of_coming)
sum_of_under_detection -= error
print("under", sum_of_under_detection)
# print('error: ', (sum_of_under_detection - (sum_of_coming + sum_of_warning)) / sum_of_under_detection)
print(sum_of_coming / sum_of_under_detection)
print(sum_of_coming, sum_of_under_detection)
