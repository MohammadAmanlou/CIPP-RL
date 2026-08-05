# راهنمای اجرای نسخه Instance-Specific

این نسخه برای هدف فعلی پروژه طراحی شده است: هر نمونه واقعی استاد یک
مسئله بهینه‌سازی مستقل است. بنابراین برای `D_14S_30P` و
`R_14S_30P` مدل جداگانه آموزش داده می‌شود. این آزمایش، ارزیابی
generalization روی داده ندیده نیست و در گزارش باید با عنوان
`instance-specific optimization` معرفی شود.

## تفاوت‌های مهم نسخه جدید

- PPO و DQN مستقیماً روی فایل واقعی همان حزب آموزش می‌بینند.
- Gurobi همان نمونه واقعی را به‌عنوان teacher حل می‌کند.
- checkpoint مربوط به D را نمی‌توان تصادفی روی R benchmark کرد.
- hyperparameterهای PPO پس از بارگذاری imitation checkpoint واقعاً
  اعمال می‌شوند.
- مدل imitation در `update=0` نیز به‌عنوان کاندید بهترین checkpoint
  ذخیره می‌شود؛ بنابراین PPO دیگر نمی‌تواند نتیجه خوب teacher را از
  بین ببرد.
- viability mask در طول مسیرهای تکراری cache می‌شود تا اجرا سریع‌تر
  شود.
- DQN هم `Greedy-1` و هم `Best-of-30` دارد.
- benchmark می‌تواند با هر ترکیبی از checkpointهای موجود اجرا شود.

## ۱. نصب و تست

در CMD ویندوز و از پوشه پروژه:

```cmd
py -3.10 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install gurobipy

python -m pytest -q
```

بررسی مجوز Gurobi:

```cmd
python -c "import gurobipy as gp; m=gp.Model('test'); x=m.addVar(lb=0); m.setObjective(x,gp.GRB.MAXIMIZE); m.addConstr(x<=1); m.optimize(); print('Status:',m.Status,'Objective:',m.ObjVal)"
```

## ۲. اولین اجرای ضروری: فقط Imitation برای D و seed=42

قبل از PPO و DQN، این فرمان را اجرا کنید. این آزمایش سریع‌ترین راه
برای بررسی درست‌بودن کل مسیر Excel → Gurobi → Imitation است.

```cmd
python -m experiments.train_imitation ^
--professor-excel CIPP-D.xls ^
--party D ^
--output-directory results/week4_instance/D/imitation_seed42 ^
--number-of-states 14 ^
--horizon 30 ^
--objective-variant professor_code ^
--solver gurobi ^
--teacher-time-limit 3600 ^
--epochs 300 ^
--batch-size 30 ^
--learning-rate 0.0003 ^
--hidden-dimension 128 ^
--normalizer-instances 8 ^
--seed 42 ^
--device cpu
```

در انتهای اجرا دو عدد مهم چاپ می‌شوند:

```text
teacher=1/1 ... objective=...
imitation_evaluation instance=D_14S_30P objective=... feasible=1
```

هدف Gurobi در Table 5 برابر `15611.6` است. اگر objective مدل
imitation به آن نزدیک نبود یا `feasible=0` بود، فعلاً هیچ آموزش
دیگری را شروع نکنید و خروجی را بررسی کنید.

## ۳. PPO معمولی روی D

بعد از موفقیت مرحله قبل:

```cmd
python -m experiments.train_ppo ^
--professor-excel CIPP-D.xls ^
--party D ^
--output-directory results/week4_instance/D/ppo_seed42 ^
--number-of-states 14 ^
--horizon 30 ^
--objective-variant professor_code ^
--updates 100 ^
--episodes-per-update 8 ^
--normalizer-instances 8 ^
--validation-interval 5 ^
--hidden-dimension 128 ^
--learning-rate 0.0002 ^
--reward-scale 0.001 ^
--discount-factor 1.0 ^
--gae-lambda 0.95 ^
--clip-epsilon 0.2 ^
--value-loss-coefficient 0.25 ^
--entropy-coefficient 0.005 ^
--gradient-clip-norm 0.5 ^
--update-epochs 4 ^
--minibatch-size 240 ^
--early-stopping-patience 8 ^
--early-stopping-min-delta 1.0 ^
--minimum-updates 40 ^
--seed 42 ^
--device auto
```

## ۴. Fine-tune مدل Imitation با PPO روی D

```cmd
python -m experiments.train_ppo ^
--professor-excel CIPP-D.xls ^
--party D ^
--initial-checkpoint results/week4_instance/D/imitation_seed42/imitation_initialization.pt ^
--output-directory results/week4_instance/D/imitation_ppo_seed42 ^
--number-of-states 14 ^
--horizon 30 ^
--objective-variant professor_code ^
--updates 50 ^
--episodes-per-update 8 ^
--normalizer-instances 8 ^
--validation-interval 5 ^
--hidden-dimension 128 ^
--learning-rate 0.00005 ^
--reward-scale 0.001 ^
--discount-factor 1.0 ^
--gae-lambda 0.95 ^
--clip-epsilon 0.15 ^
--value-loss-coefficient 0.25 ^
--entropy-coefficient 0.001 ^
--gradient-clip-norm 0.5 ^
--update-epochs 2 ^
--minibatch-size 240 ^
--early-stopping-patience 6 ^
--early-stopping-min-delta 1.0 ^
--minimum-updates 20 ^
--seed 42 ^
--device auto
```

در این اجرا ابتدا نتیجه زیر چاپ می‌شود:

```text
update=0 initial_validation_mean=...
```

این همان policy تقلیدی قبل از PPO است و اگر بهترین نتیجه باقی بماند،
`checkpoint_best.pt` همان update صفر را نگه می‌دارد.

## ۵. DQN روی D

```cmd
python -m experiments.train_dqn ^
--professor-excel CIPP-D.xls ^
--party D ^
--output-directory results/week4_instance/D/dqn_seed42 ^
--number-of-states 14 ^
--horizon 30 ^
--objective-variant professor_code ^
--episodes 300 ^
--normalizer-episodes 8 ^
--hidden-dimension 128 ^
--learning-rate 0.0002 ^
--reward-scale 0.001 ^
--batch-size 128 ^
--warmup-transitions 512 ^
--target-sync-steps 300 ^
--epsilon-start 1.0 ^
--epsilon-end 0.05 ^
--epsilon-decay-fraction 0.70 ^
--validation-interval 10 ^
--early-stopping-patience 10 ^
--minimum-episodes 100 ^
--seed 42 ^
--device auto
```

## ۶. Benchmark کامل D

```cmd
python -m experiments.benchmark_week4 ^
--ppo-checkpoint results/week4_instance/D/ppo_seed42/checkpoint_best.pt ^
--imitation-ppo-checkpoint results/week4_instance/D/imitation_ppo_seed42/checkpoint_best.pt ^
--dqn-checkpoint results/week4_instance/D/dqn_seed42/checkpoint_best.pt ^
--democrat-excel CIPP-D.xls ^
--parties D ^
--number-of-states 14 ^
--horizon 30 ^
--objective-variant professor_code ^
--rollouts 30 ^
--dqn-rollout-epsilon 0.10 ^
--output-directory results/week4_instance/D/benchmark_seed42 ^
--seed 42 ^
--device cpu
```

خروجی شامل این روش‌هاست:

- Greedy
- DQN-Greedy-1
- DQN-Best-of-30
- PPO-Greedy-1
- PPO-Best-of-30
- Imitation+PPO-Greedy-1
- Imitation+PPO-Best-of-30
- Gurobi-Table5

## ۷. تکرار برای R

بعد از کامل‌شدن D با seed 42، همان چهار مرحله را با این تغییرها اجرا
کنید:

```text
CIPP-D.xls  → CIPP-R.xls
--party D   → --party R
.../D/...   → .../R/...
```

مرجع Table 5 برای `R_14S_30P` برابر `14745.4` است. هرگز checkpoint
مربوط به D را در benchmark مربوط به R استفاده نکنید.

## ۸. seedهای 43 و 44

تا وقتی اجرای کامل seed 42 برای هر دو حزب نتیجه قابل‌قبولی نداده،
seedهای دیگر را شروع نکنید. پس از تأیید seed 42، مسیر خروجی و آرگومان
seed را به 43 و سپس 44 تغییر دهید.

## ۹. اجرای مجدد Gurobi در benchmark

جدول پیش‌فرض از مقدار منتشرشده Table 5 استفاده می‌کند. برای بازتولید
مستقل، این آرگومان‌ها را به فرمان benchmark اضافه کنید:

```cmd
--run-exact ^
--exact-solver gurobi ^
--exact-time-limit 3600
```

## ۱۰. فایل‌های نهایی

از هر آموزش:

```text
checkpoint_best.pt
checkpoint_last.pt
experiment_config.json
training_history.json
learning_curve.png
```

از Imitation:

```text
imitation_initialization.pt
imitation_results.json
exact_solver/*.gurobi.log
```

از benchmark:

```text
comparison_table.csv
comparison_table.md
comparison_table.tex
benchmark_details.json
```

پس از پایان سه seed و هر دو حزب:

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'results\week4_instance\*' -DestinationPath 'Week4_InstanceSpecific_Results.zip' -Force"
```
