# Them dau cho tieng Viet khong dau


# Overview

Đây là một baseline cho bài toán thêm dấu tiếng Việt sử dụng mô hình Transformer.

Phần lớn code cho mô hình được lấy từ repo này của Kyubyong Park: https://github.com/Kyubyong/transformer .

Ở đây có một số thay đổi, cụ thể:

- Mình bỏ phần encoder đi, chỉ giữ lại 1 nửa decoder. Do bài toán này là map 1-1 từ input về output nên mô hình sẽ không cần autoregressive deocoding nữa.
- Không sử dụng label smoothing cho hàm loss vì sẽ làm model hội tụ chậm hơn.

# Trainng 

Để chuẩn bị training bạn cần chuẩn bị các file sau:

## Tập training

Cần 2 file ```train.src``` và ```train.tgt```. Trong đó file ```train.tgt``` chính là file train mà ban tổ chức đã cho, và file ```train.src``` là sau khi đã bỏ dấu.

## Tập test


## Tập từ điển (vocabulary)



## Quá trình training/test

Các hyperparameter được lưu lại trong file ```hyperparams.py```. Để training bạn chỉ cần chạy:

```python train.py```

Để sinh file kết quả chạy:

```python eval.py```

Kết quả sẽ sinh ra trong file ```results/logdir.txt``` .

