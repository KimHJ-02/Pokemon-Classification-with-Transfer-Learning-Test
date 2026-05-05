import os
import shutil
import kagglehub
import zipfile


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
SPLIT_NAMES = ('train', 'val', 'valid', 'validation', 'test')


def _collect_images_by_class(root_path):
    by_class = {}
    for dirpath, _, filenames in os.walk(root_path):
        images = [
            os.path.join(dirpath, fn)
            for fn in filenames
            if fn.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if not images:
            continue
        cls = os.path.basename(dirpath)
        if cls.lower() in SPLIT_NAMES:
            continue
        by_class.setdefault(cls, [])
        by_class[cls].extend(sorted(images))
    return by_class


def _copy_split_folder(src_root, split_name, dst_root):
    src_split = os.path.join(src_root, split_name)
    if not os.path.isdir(src_split):
        return False
    dst_split = os.path.join(dst_root, split_name)
    os.makedirs(dst_split, exist_ok=True)
    copied = 0
    for cls in sorted(os.listdir(src_split)):
        cls_src = os.path.join(src_split, cls)
        if not os.path.isdir(cls_src):
            continue
        images = [
            fn for fn in os.listdir(cls_src)
            if fn.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if not images:
            continue
        cls_dst = os.path.join(dst_split, cls)
        os.makedirs(cls_dst, exist_ok=True)
        for fn in images:
            shutil.copy2(os.path.join(cls_src, fn), os.path.join(cls_dst, fn))
            copied += 1
    return copied > 0


def _copy_train_val_split(images_by_class, target_dir, val_ratio=0.2):
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for cls, images in images_by_class.items():
        split_idx = max(1, int(len(images) * (1.0 - val_ratio)))
        train_images = images[:split_idx]
        val_images = images[split_idx:] or images[-1:]

        cls_train_dir = os.path.join(train_dir, cls)
        cls_val_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_train_dir, exist_ok=True)
        os.makedirs(cls_val_dir, exist_ok=True)

        for src in train_images:
            shutil.copy2(src, os.path.join(cls_train_dir, os.path.basename(src)))
        for src in val_images:
            shutil.copy2(src, os.path.join(cls_val_dir, os.path.basename(src)))


def _prepare_dataset(source_path, target_dir):
    # 1) If source already has train/val structure, keep it.
    copied_train = _copy_split_folder(source_path, 'train', target_dir)
    copied_val = _copy_split_folder(source_path, 'val', target_dir)
    if not copied_val:
        copied_val = _copy_split_folder(source_path, 'valid', target_dir)
    if copied_train and copied_val:
        return True, 'Copied existing train/val folders.'

    # 2) Otherwise collect class folders recursively and make split.
    images_by_class = _collect_images_by_class(source_path)
    if not images_by_class:
        return False, 'No class folders with images found.'
    _copy_train_val_split(images_by_class, target_dir)
    return True, 'Prepared train/val split from class folders.'


def main():
    # dataset identifier from kaggle
    ds = "lantian773030/pokemonclassification"
    print("Downloading dataset:", ds)
    path = kagglehub.dataset_download(ds)
    print("Downloaded to:", path)
    # if it's a zip file, extract to data/
    target_dir = 'data'
    os.makedirs(target_dir, exist_ok=True)
    if path.lower().endswith('.zip'):
        print('Extracting...')
        extracted_dir = os.path.join(target_dir, '_raw')
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(extracted_dir)
        print('Extracted to', extracted_dir)
        ok, msg = _prepare_dataset(extracted_dir, target_dir)
        print(msg)
        if ok:
            print('Prepared dataset in', target_dir)
    else:
        ok, msg = _prepare_dataset(path, target_dir) if os.path.isdir(path) else (False, 'Downloaded path is not a directory.')
        print(msg)
        if ok:
            print('Prepared dataset in', target_dir)
        else:
            print('Failed to prepare dataset from downloaded source.')

if __name__=='__main__':
    main()
