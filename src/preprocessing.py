import os, pathlib, shutil

original_dir = pathlib.Path("Data\\Pictures")
new_base_dir = pathlib.Path("Data\\Sets")

def make_subset():
    for category in ("Angry", "Fear", "Happy", "Sad", "Suprise"):
        print(f"current category: {category}")
        dir = original_dir / category
        datasize = 0
        fnames = []
        for root, dirnames, filenames in os.walk(dir):
            for file in filenames:
                datasize += 1
                fnames.append(file)

        for step, subset in enumerate(["train", "validation", "test"]):
            print(f"current subset: {subset}")
            indicies = [0, round(datasize * 0.5), round(datasize*0.75), datasize]
            dir = new_base_dir / subset / category
            os.makedirs(dir)

            fnames_now = fnames[indicies[step] : indicies[step+1] ]

            for fname in fnames_now:
                shutil.copyfile(
                    src=original_dir / category / fname,
                    dst=dir / fname
                )
            
make_subset()