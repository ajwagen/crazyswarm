import numpy as np
import ast

# def convert_string_to_list(string):
#   """Converts a string of numbers into a list of numbers.

#   Args:
#     string: The string of numbers to convert.

#   Returns:
#     A list of numbers.
#   """

#   list_of_numbers = []
#   for number in string.strip().split():
#     if number.isdigit():
#       list_of_numbers.append(int(number))

#   return list_of_numbers


# if __name__ == "__main__":
#   string = "[0  1  2]"
#   list_of_numbers = convert_string_to_list(string)
#   print(list_of_numbers)

sim = []
real = []

def convert_string_to_list(string):
  """Converts a string of numbers into a list of numbers.

  Args:
    string: The string of numbers to convert.

  Returns:
    A list of numbers.
  """

  list_of_numbers = []
  for number in string.split():
    if number.isdigit():
      list_of_numbers.append(int(number))

  return list_of_numbers

with open('out.txt') as f:
    lines = f.readlines()

    for line in lines:
        if "sim" in line:
            line = line.strip('\n')
            line = line.strip('sim RMSE :  ')
            line = ast.literal_eval(line)
            sim.append(line)
        if "real" in line:
            line = line.strip('\n')
            line = line.strip('real RMSE :  ')
            line = ast.literal_eval(line)
            real.append(line)

sim_mean = np.mean(sim, axis=0)
for i in sim_mean:
   print(i)
# print(np.mean(sim, axis=0))
print(np.mean(sim))
print("")
real_mean = np.mean(real, axis=0)
for i in real_mean:
   print(i)
# print(np.mean(real, axis=0))
print(np.mean(real))
