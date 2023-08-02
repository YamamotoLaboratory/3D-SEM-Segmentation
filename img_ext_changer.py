import sys, argparse, re, os

from PIL import Image

sys.dont_write_bytecode = True

from log import load_logging_config

def png2bmp(args, logger):
    try:
        img = Image.open(args.image)
        new_img_path = re.findall(r'(.+)\{}'.format(os.path.splitext(os.path.basename(args.image))[-1]), args.image)[0]+'.{}'.format(args.ext)
        img.save(new_img_path)
        logger.info('画像ファイルの変換が完了しました。{}に保存されています。'.format(new_img_path))
    except Exception as e:
        logger.error('画像ファイルの読み込みに失敗しました。')
        logger.debug(e, stack_info=True)
        sys.exit('プログラムを終了します。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='画像の拡張子を変更するプログラムになります。')
    parser.add_argument('--debug', action='store_true', help='debugモードでプログラムを実行します。')
    parser.add_argument('-i', '--image', help='変換したい画像ファイルを指定します。', required=True)
    parser.add_argument('-e', '--ext', help='変換したい画像ファイルをの拡張子を指定します。', choices=['png', 'bmp'],  required=True)

    args = parser.parse_args()

    logger = load_logging_config((lambda args: 'debug_logger' if args.debug else __name__)(args))

    png2bmp(args, logger)

