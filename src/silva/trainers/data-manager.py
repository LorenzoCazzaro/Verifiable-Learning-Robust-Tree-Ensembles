import os
import sys
import zipfile

# Compresses a single file
def compress(file_path, keep_original = False):
  base_name = os.path.basename(file_path)
  print("Compressing " + file_path + "...")
  with zipfile.ZipFile(file_path + '.zip', 'w', zipfile.ZIP_DEFLATED) as zip_ref:
    zip_ref.write(file_path, base_name)
  if not keep_original:
    os.remove(file_path)


# Recursively compresses every file inside a directory
def compress_all(directory = './domains'):
  for f in os.listdir(directory):
    file_path = directory + '/' + f
    if os.path.isdir(file_path):
      compress_all(file_path)
    elif os.path.isfile(file_path) and os.path.splitext(file_path)[1] == '.csv':
      compress(file_path)


#Decompresses a single zip file
def decompress(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    directory = os.path.dirname(file_path)
    print("Decompressing " + file_path + "...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
      zip_ref.extractall(directory)


# Recursively decompress every archive in a directory
def decompress_all(directory = './domains'):
  for f in os.listdir(directory):
    file_path = directory + '/' + f
    if os.path.isdir(file_path):
      decompress_all(file_path)
    elif os.path.isfile(file_path) and os.path.splitext(file_path)[1] == '.zip':
      decompress(file_path)


# Recursively removes non-archive files
def clear_all(directory = './domains'):
  for f in os.listdir(directory):
    file_path = directory + '/' + f
    if os.path.isdir(file_path):
      clear_all(file_path)
    elif os.path.isfile(file_path) and os.path.splitext(file_path)[1] != '.zip':
      print("Clearing " + file_path + "...")
      os.remove(file_path)


# Prints usage
def print_usage():
  print("Usage:\n\t" + sys.argv[0] + " compress|decompress|clear <directory>\n")



#######################################################################
# Program entry point

# Input check
if len(sys.argv) < 3:
  print_usage()
  sys.exit()

# Reads mode
mode = sys.argv[1]

# Executes operation
if mode == 'compress':
  compress_all(sys.argv[2])
elif mode == 'decompress':
  decompress_all(sys.argv[2])
elif mode == 'clear':
  clear_all(sys.argv[2])
else:
  print_usage()
  sys.exit()
