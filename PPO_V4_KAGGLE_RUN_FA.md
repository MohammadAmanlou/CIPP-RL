# راهنمای اجرای PPO Suite v4 روی Kaggle

> قرارداد قطعی: در `D_14S_30P` عدد 14 تعداد state/locationهای قابل بازدید است. Idle یک action مستقل با اندیس صفر است. بنابراین `instance.n=14` و `num_actions=15`. ردیف صفر فایل Excel فقط منبع تعریف Idle است و دوباره به عنوان location وارد مدل نمی‌شود.

## قرارداد درست D_14S_30P
در این نسخه، `14S` یعنی **۱۴ state قابل بازدید**:
- action صفر: `Idle/Rest` با reward و cost صفر
- ۱۴ action بازدید: ۱۴ state/location واقعی
- `30P`: افق ۳۰ روزه

ردیف Idle در فایل داده فقط اعتبارسنجی می‌شود و دوباره به‌عنوان location وارد مدل نمی‌شود.

## قابلیت‌های نسخه v4
- لاگ دوره‌ای با `--log-every-episodes`
- Early stopping کامل برای Construction PPO و RL Improvement
- validation ثابت و قابل بازتولید
- انتخاب checkpoint بر اساس مدل جاری، نه archive صعودی
- rolloutهای batch شده برای استفاده بهتر از GPU
- beam search batch شده
- `--benchmark-only` واقعی و سخت‌گیرانه
- زمان کل Improvement شامل ساخت start و خود improvement

## 1. آماده‌سازی Kaggle
Dataset پروژه را اضافه کن و سپس:

```python
%cd /kaggle/working
```

اگر فایل ZIP است:

```bash
!rm -rf CIPP-RL-PPO-Suite-v4 && mkdir -p CIPP-RL-PPO-Suite-v4 && unzip -q /kaggle/input/NAME-OF-DATASET/CIPP-RL-PPO-Suite-v4.zip -d CIPP-RL-PPO-Suite-v4
```

```python
%cd /kaggle/working/CIPP-RL-PPO-Suite-v4/CIPP-RL-PPO-Suite
```

```bash
!python -m pip install -q -r requirements.txt
```

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 2. تست کامل

```bash
!python -m pytest -q
```

خروجی اعتبارسنجی این نسخه:

```text
75 passed, 1 skipped
```

## 3. Smoke Run

```bash
!python -m experiments.run_ppo_suite --instances D_14S_30P --profile smoke --methods stable_mlp attention hierarchical hacipp hacipp_rl_improve --log-every-episodes 2 --device auto --seed 42 --data-directory . --output-directory /kaggle/working/results/ppo_v4_smoke
```

## 4. اجرای اصلی همه مدل‌ها برای Ablation

```bash
!python -m experiments.run_ppo_suite --instances D_14S_30P --profile full --methods stable_mlp attention hierarchical hacipp hacipp_rl_improve --log-every-episodes 128 --device auto --seed 42 --data-directory . --output-directory /kaggle/working/results/ppo_v4_full
```

در پروفایل `full`:
- Construction: حداکثر ۴۰۰ update، هر update برابر ۶۴ episode
- validation هر ۲۰ update
- early stopping بعد از ۶ validation بدون بهبود معنادار
- warmup برابر ۱۲۰ update
- Improvement: حداکثر ۲۰۰ update
- validation هر ۱۰ update
- early stopping بعد از ۸ validation بدون بهبود
- validation startها ثابت هستند

## 5. اجرای سریع‌تر فقط مدل نهایی

```bash
!python -m experiments.run_ppo_suite --instances D_14S_30P --profile full --methods hacipp hacipp_rl_improve --log-every-episodes 128 --device auto --seed 42 --data-directory . --output-directory /kaggle/working/results/ppo_v4_final
```

## 6. تغییر Early Stopping از خط فرمان

```bash
!python -m experiments.run_ppo_suite --instances D_14S_30P --profile full --methods hacipp hacipp_rl_improve --construction-early-stopping-patience 6 --construction-early-stopping-min-delta 1.0 --construction-early-stopping-warmup-updates 120 --improvement-early-stopping-patience 8 --improvement-early-stopping-min-delta 1.0 --improvement-early-stopping-warmup-updates 60 --log-every-episodes 128 --device auto --seed 42 --data-directory . --output-directory /kaggle/working/results/ppo_v4_final
```

برای غیرفعال‌کردن Early Stopping مقدار patience را صفر قرار بده.

## 7. Resume
اگر checkpoint وجود داشته باشد، مدل بارگذاری می‌شود؛ اگر وجود نداشته باشد، آموزش شروع می‌شود:

```bash
!python -m experiments.run_ppo_suite --instances D_14S_30P --profile full --methods hacipp hacipp_rl_improve --resume --log-every-episodes 128 --device auto --seed 42 --data-directory . --output-directory /kaggle/working/results/ppo_v4_final
```

## 8. Benchmark-only واقعی
این حالت هرگز train نمی‌کند و در صورت نبود checkpoint خطا می‌دهد:

```bash
!python -m experiments.run_ppo_suite --instances D_14S_30P --profile full --methods hacipp hacipp_rl_improve --benchmark-only --device auto --seed 42 --data-directory . --output-directory /kaggle/working/results/ppo_v4_final
```

## 9. Benchmark با Gurobi
در محیطی که Gurobi و license فعال است:

```bash
!python -m experiments.run_ppo_suite --instances D_14S_30P --profile full --methods hacipp hacipp_rl_improve --benchmark-only --run-gurobi --gurobi-time-limit 3600 --device auto --seed 42 --data-directory . --output-directory /kaggle/working/results/ppo_v4_final
```

## 10. خروجی‌ها
برای هر مدل:
- `checkpoint_best.pt`
- `checkpoint_last.pt`
- `training_summary.json`
- `history.json`
- `history.csv`
- `elite_archive.json` برای Construction
- `fixed_validation_starts.json` برای Improvement

خروجی مشترک:
- `leaderboard.csv`
- `leaderboard.json`
- `leaderboard.md`
- `run_manifest.json`

## 11. ZIP کردن خروجی Kaggle

```bash
!cd /kaggle/working && zip -qr ppo_v4_results.zip results/ppo_v4_full
```

## نکات زمان‌گیری
- `runtime_seconds` برای Improvement شامل زمان تولید start و زمان RL improvement است.
- `start_generation_seconds` و `improvement_seconds` جدا نیز ذخیره می‌شوند.
- training time هزینه offline است و باید جدا از online inference time گزارش شود.