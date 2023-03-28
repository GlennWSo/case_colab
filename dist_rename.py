from os import walk, rename
from os.path import splitext, join

folder = "dist"
filenames = next(walk(folder), (None, None, []))[2]  # [] if no file

wheel_name = join(
    folder, [name for name in filenames if splitext(name)[1] == ".whl"][0]
)
new_name = wheel_name.replace("linux", "manylinux2014")
print(f"renaming file: { wheel_name}")
print(f"new name: {new_name}")

rename(wheel_name, new_name)
