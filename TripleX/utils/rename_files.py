import os


def rename_pairs(directory):
    # Get a list of all filenames in the directory
    all_files = os.listdir(directory)

    # Collect base names that have both .mp4 and .txt files.
    base_names = set()
    for filename in all_files:
        if filename.endswith('.mp4') or filename.endswith('.txt'):
            # Remove the extension and add to set
            base = os.path.splitext(filename)[0]
            base_names.add(base)

    # Make a sorted list so ordering is predictable.
    base_names = sorted(base_names)

    counter = 0
    for base in base_names:
        mp4_file = os.path.join(directory, base + '.mp4')
        txt_file = os.path.join(directory, base + '.txt')

        # Check both files exist, if not skip this base name.
        if not (os.path.exists(mp4_file) and os.path.exists(txt_file)):
            print(f"Skipping {base}: both .mp4 and .txt files are not present.")
            continue

        new_mp4_file = os.path.join(directory, str(counter) + '.mp4')
        new_txt_file = os.path.join(directory, str(counter) + '.txt')

        # Rename files
        print(f"Renaming {mp4_file} -> {new_mp4_file}")
        print(f"Renaming {txt_file} -> {new_txt_file}")
        os.rename(mp4_file, new_mp4_file)
        os.rename(txt_file, new_txt_file)

        counter += 1


if __name__ == '__main__':
    # Change '.' to your target directory if necessary.
    target_directory = 'data/captioned'
    rename_pairs(target_directory)