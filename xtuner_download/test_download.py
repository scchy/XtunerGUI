from download_model import xtunerModelDownload
from download_dataset import xtunerDataDownload
from tqdm.auto import tqdm
import time 

def main():
    print('>>>>>>>> Start xtunerModelDownload')
    model_name = 'internlm/internlm-chat-7b'
    d_model = xtunerModelDownload(
        model_name,
        out_path='/root/tmp/download_model',
        tqdm_class=tqdm
    )
    d_model.auto_download()
    print('>>>>>>>> Start xtunerDataDownload')
    data_name = 'shibing624/medical'
    d_data = xtunerDataDownload(
        data_name,
        out_path='/root/tmp/download_data',
        tqdm_class=tqdm,
        retry_times=0
    )
    d_data.auto_download()
    time.sleep(60)
    d_data.break_download()
    d_model.break_download()
    print('Yes')
    

if __name__ == '__main__':
    main()
