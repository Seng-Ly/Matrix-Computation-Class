
import matplotlib.pyplot as plt
import numpy as np

file_data   = "mnist_test.csv"
handle_file = open(file_data, "r")
data        = handle_file.readlines()
handle_file.close()

size_row    = 28    # height of the image
size_col    = 28    # width of the image

num_image   = len(data)
count       = 0     # count for the number of images

#
# make a matrix each column of which represents an images in a vector form
#
list_image  = np.empty((size_row * size_col, num_image), dtype=float)
list_label  = np.empty(num_image, dtype=int)

for line in data:

    line_data   = line.split(',')
    label       = line_data[0]
    im_vector   = np.asfarray(line_data[1:])

    list_label[count]       = label
    list_image[:, count]    = im_vector

    count += 1

#
# plot first 100 images out of 10,000 with their labels
#
# f1 = plt.figure(figsize=(10, 10))
#
# for i in range(100):
#
#     label       = list_label[i]
#     im_vector   = list_image[:, i]
#     im_matrix   = im_vector.reshape((size_row, size_col))
#
#     plt.subplot(10, 10, i+1)
#     plt.title(label)
#     plt.imshow(im_matrix, cmap='Greys', interpolation='None')
#
#     frame   = plt.gca()
#     frame.axes.get_xaxis().set_visible(False)
#     frame.axes.get_yaxis().set_visible(False)
#
# plt.show()

# increase = 0
# #
# # for i in range(100):
# #
# #     if list_label[i] == increase:
# #
# #         label       = list_label[i]
# #         im_vector   = list_image[:, i]
# #         im_matrix   = im_vector.reshape((size_row, size_col))
# #
# #         plt.subplot(2, 5, increase + 1)
# #         plt.title(label)
# #         plt.imshow(im_matrix, cmap='Greys', interpolation='None')
# #
# #         frame   = plt.gca()
# #         frame.axes.get_xaxis().set_visible(False)
# #         frame.axes.get_yaxis().set_visible(False)
# #         increase += 1
# #     if increase == 10: break
# #
# # plt.show()


# col_avg = []
#
# num_avg = 10
# for j in range(10):
#     increase = 0
#     tol = 0
#     for i in range(num_avg*10):
#         if increase == num_avg: break
#         if list_label[i] == j:
#             tol += list_image[:,i]
#             increase += 1
#     col_avg.append(tol/num_avg)
#
# plt.figure(figsize=(10, 10))
# for i in range(10):
#     im_vector   = col_avg[i]
#     im_matrix   = im_vector.reshape((size_row, size_col))
#     plt.subplot(2, 5, i + 1)
#     plt.title(i)
#     plt.imshow(im_matrix, cmap='Greys', interpolation='None')
# plt.show()

# xx = [1,2,3]
# xx = np.array(xx)
#
# ww = np.random.normal(loc=0, scale=1, size=[3,2])
#
# print(xx.dot(ww))


im_vector   = list_image[:, 0]


x = im_vector
x = np.array(x)
w = np.random.normal(loc=0, scale=1, size=[len(x), 10])

y = x.dot(w)

print(y)