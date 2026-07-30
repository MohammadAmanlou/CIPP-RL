# راهنمای اجرای مرحله ۴

## هدف

یک DQN را روی تعداد زیادی instance کالیبره‌شده آموزش می‌دهیم و همان checkpoint
ثابت را به دو شکل ارزیابی می‌کنیم:

1. `DQN-Greedy`: یک مسیر کامل، بدون برگشت و بدون آموزش در تست.
2. `DQN-Backtracking-30`: همان شبکه، با برگشت به prefixهای قبلی و بررسی حداکثر
   ۳۰ مسیر کامل feasible.

پس دو ردیف DQN به معنی دو شبکه جدا نیستند؛ تفاوت فقط در decoder و بودجه جست‌وجو
است. در تست هیچ `backward`، optimizer update یا replay-buffer update انجام
نمی‌شود.

## نکته مهم درباره کوچک‌ترین instance

کد استاد با `Cities=16` شامل اندیس صفر `Rest` و ۱۵ location واقعی است. مقاله
نسخه ۱۴-state را گزارش می‌کند. برای جلوگیری از گزارش اشتباه دو حالت داریم:

```text
--instance-mode supplied-code  -> D/R_15S_30P_CODE
--instance-mode paper-14       -> D/R_14S_30P_PAPER_SHAPE
```

تا وقتی subset دقیق ۱۴-state توسط استاد تأیید نشده، عدد حالت ۱۵-state را با جدول
۱۴-state مقاله یکی ندانید.

## نصب و تست

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.run_week2_validation --episodes 1000
```

## آموزش اصلی

```bash
python -m experiments.train_dqn_real_matched \
  --instance-mode supplied-code \
  --episodes 20000 \
  --seeds 0 1 2 3 4 \
  --output-dir checkpoints/dqn_real_matched_code
```

آموزش روی جریان instanceهای synthetic انجام می‌شود که از profileهای واقعی D و R
ترکیب، scale، jitter و permutation می‌گیرند. خود instance تست به replay buffer
فرستاده نمی‌شود. بهترین checkpoint با validation ثابت انتخاب می‌شود.

## Benchmark با Gurobi

```bash
python -m experiments.benchmark_smallest_real_instance \
  --party D \
  --instance-mode supplied-code \
  --checkpoints \
    checkpoints/dqn_real_matched_code/seed_0/best.pt \
    checkpoints/dqn_real_matched_code/seed_1/best.pt \
    checkpoints/dqn_real_matched_code/seed_2/best.pt \
    checkpoints/dqn_real_matched_code/seed_3/best.pt \
    checkpoints/dqn_real_matched_code/seed_4/best.pt \
  --rollouts 30 \
  --random-runs 30 \
  --gurobi-time-limit 3600 \
  --output-dir results/smallest_real_D
```

برای حزب R فقط `--party R` را عوض کنید.

## معنی Gap

اگر Gurobi به `OPTIMAL` برسد، reference یک optimum اثبات‌شده است. اگر time limit
تمام شود، reference فقط incumbent گوروبی است. جدول همچنین `Gurobi MIP Gap` را در
ستون جداگانه ثبت می‌کند.

## خروجی جدول

- `comparison_table.csv`
- `comparison_table.md`
- `comparison_table.tex`
- `benchmark_details.json`
- `results/comparison_table_template.xlsx`

ستون‌های مهم شامل BFS، میانگین، انحراف معیار، best/mean gap، زمان کل، تعداد
rollout، feasibility، idle days و unique locations هستند.
