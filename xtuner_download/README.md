
## 模型下载

核心流程
```mermaid
flowchart LR

IN1(model_name) --> A
IN2(out_path) --> A
IN3(tqdm_class) --> A
subgraph xtunerModelDownload
A(initial) --> B(获取模型信息:文件数量、文件大小)

DM(起用下载线程：X-model-download)
DP(起用进度线程：X-model-progress)
clear(文件清除)
DP -->|检查下载进度|DM
end

B --> C
B --> Break
C -->|Start线程|DM
C -->|Start线程|DP --> grP
subgraph gr-client

C(模型下载)
grP(进度条显示)

Break(下载中断)
end

Break -->|kill线程|DM
Break -->|kill线程|DP
Break --> clear

DM --> hf
subgraph download
modelscope(modelscope-download)
hf(huggingface-download)
openxlab(openxlab-download)

hf--> check1{下载成功?}-->|否|modelscope--> check2{下载成功?}-->|否|openxlab

check1 -->|是|finshed
check2 -->|是|finshed
end
```

- example:
```python
from download_model import xtunerModelDownload
from tqdm.auto import tqdm
import time 

model_name = 'internlm/internlm-chat-7b'
d_model = xtunerModelDownload(
    model_name,
    out_path='/root/tmp/download_model',
    tqdm_class=tqdm
)
d_model.auto_download()
time.sleep(60)
d_model.break_download()
print('Yes')
```

## 数据下载


核心流程
```mermaid
flowchart LR

IN1(data_name) --> A
IN2(out_path) --> A
IN3(tqdm_class) --> A
subgraph xtunerModelDownload
A(initial) --> B(获取数据信息:文件数量、文件大小)

DM(起用下载线程：X-dataset-download)
DP(起用进度线程：X-dataset-progress)
clear(文件清除)
DP -->|检查下载进度|DM
end

B --> C
B --> Break
C -->|Start线程|DM
C -->|Start线程|DP --> grP
subgraph gr-client

C(数据下载)
grP(进度条显示)

Break(下载中断)
end

Break -->|kill线程|DM
Break -->|kill线程|DP
Break --> clear

DM --> hf
subgraph download
hf(huggingface-download)

hf--> check1{下载成功?}-->|否|hf-->|Retry-n times|check1

check1 -->|是|finshed
end
```

- example:
```python
from download_dataset import xtunerDataDownload
from tqdm.auto import tqdm
import time 

data_name = 'shibing624/medical'
d_data = xtunerDataDownload(
    data_name,
    out_path='/root/tmp/download_data',
    tqdm_class=tqdm,
    retry_times=1
)
d_data.auto_download()
time.sleep(60)
d_data.break_download()
print('Yes')
```


