# هشدار نسخه
این راهنما مربوط به نسخه قدیمی است. برای نسخه فعلی از `PPO_V4_KAGGLE_RUN_FA.md` استفاده کنید. در نسخه v4، `D_14S_30P` یعنی ۱۴ state قابل بازدید به‌علاوه یک action مستقل Idle؛ در مجموع ۱۵ action.

# راهنمای کامل اجرای PPOهای CIPP در Kaggle

این نسخه برای اجرای مستقیم در Kaggle آماده شده است. روش اصلی کاملاً RL است:

```text
Hierarchical Attention PPO construction
→ Best-of-K / policy-only SGBS
→ PPO-based neural neighborhood improvement (حداکثر ۳۰ حرکت)
```

Gurobi در تصمیم‌های عامل، repair، label یا reward نقشی ندارد و فقط benchmark اختیاری است.

## ۱. تنظیم Notebook

در Kaggle یک Notebook جدید بسازید و از بخش `Settings`:

- Accelerator را روی `GPU T4 x2` یا `GPU P100` بگذارید؛
- Internet را برای نصب dependencyها روشن کنید؛
- فایل `CIPP-RL-PPO-Suite.zip` را به‌صورت Dataset به Notebook اضافه کنید.

## ۲. استخراج پروژه

سلول اول:

```bash
%%bash
set -e
PROJECT_ZIP=$(find /kaggle/input -name 'CIPP-RL-PPO-Suite.zip' | head -n 1)
test -n "$PROJECT_ZIP"
rm -rf /kaggle/working/CIPP-RL-PPO-Suite
unzip -q "$PROJECT_ZIP" -d /kaggle/working
cd /kaggle/working/CIPP-RL-PPO-Suite
python -m pip install -q -r requirements.txt
```

## ۳. تست اجباری قبل از ران اصلی

```bash
%%bash
set -e
cd /kaggle/working/CIPP-RL-PPO-Suite
python -m pytest -q
python -m experiments.run_ppo_suite \
  --instances D_14S_30P \
  --profile smoke \
  --methods stable_mlp attention hierarchical hacipp hacipp_rl_improve \
  --device auto \
  --output-directory results/kaggle_smoke
```

انتهای خروجی باید `completed=14 rows` را نشان دهد. نتایج smoke برای مقاله نیستند، چون هر مدل فقط دو update می‌گیرد.

## ۴. Ablation کامل روی نمونه کوچک

این دستور تمام مدل‌ها را روی D و R اجرا می‌کند:

```bash
%%bash
set -e
cd /kaggle/working/CIPP-RL-PPO-Suite
python -m experiments.run_ppo_suite \
  --instances D_14S_30P R_14S_30P \
  --profile full \
  --methods stable_mlp attention hierarchical hacipp hacipp_rl_improve \
  --device auto \
  --seed 42 \
  --output-directory results/ablation_14S_30P
```

برای گزارش چند seed، همین دستور را جداگانه با `--seed 0`، `1`، `2`، `3` و `4` و output directory متفاوت اجرا کنید.

## ۵. ران اصلی instanceهای سخت D

```bash
%%bash
set -e
cd /kaggle/working/CIPP-RL-PPO-Suite
python -m experiments.run_ppo_suite \
  --instances D_30S_75P D_30S_90P D_51S_75P D_51S_90P \
  --profile full \
  --methods hacipp hacipp_rl_improve \
  --construction-updates 800 \
  --episodes-per-update 64 \
  --improvement-updates 300 \
  --final-rollouts 256 \
  --beam-width 64 \
  --device auto \
  --seed 42 \
  --output-directory results/hard_D
```

## ۶. ران اصلی instanceهای سخت R

```bash
%%bash
set -e
cd /kaggle/working/CIPP-RL-PPO-Suite
python -m experiments.run_ppo_suite \
  --instances R_30S_75P R_30S_90P R_51S_75P R_51S_90P \
  --profile full \
  --methods hacipp hacipp_rl_improve \
  --construction-updates 800 \
  --episodes-per-update 64 \
  --improvement-updates 300 \
  --final-rollouts 256 \
  --beam-width 64 \
  --device auto \
  --seed 42 \
  --output-directory results/hard_R
```

برای جلوگیری از تمام‌شدن زمان Kaggle بهتر است D و R را در دو Notebook/Session جدا اجرا کنید. همچنین می‌توانید `30S` و `51S` را جدا کنید.

## ۷. ادامه‌دادن یک اجرای قطع‌شده

اگر checkpointها هنوز در `/kaggle/working` هستند، همان دستور را با `--resume` اجرا کنید. مدل‌هایی که `checkpoint_best.pt` دارند دوباره train نمی‌شوند:

```text
--resume
```

### آزمایش Efficient Active Search

ابتدا HACIPP را با حالت عادی `full` train کنید. سپس فقط برای همان instance و
همان معماری، checkpoint را به EAS بدهید تا encoder ثابت و adapter/decoder روی
instance سازگار شود:

```bash
python -m experiments.run_ppo_suite \
  --instances D_51S_90P \
  --methods hacipp \
  --profile quick \
  --active-search-mode eas \
  --initial-checkpoint results/hard_D/D_51S_90P/hacipp/seed_42/checkpoint_best.pt \
  --output-directory results/eas_D_51S_90P
```

اجرای EAS از وزن تصادفی عمداً مجاز نیست؛ چون EAS باید از یک backbone از قبل
آموزش‌دیده شروع شود.

## ۸. خروجی‌ها

در هر output directory این فایل‌ها ساخته می‌شوند:

```text
leaderboard.csv
leaderboard.md
leaderboard.json
run_manifest.json
INSTANCE/METHOD/seed_k/checkpoint_best.pt
INSTANCE/METHOD/seed_k/checkpoint_last.pt
INSTANCE/METHOD/seed_k/history.csv
INSTANCE/METHOD/seed_k/history.json
INSTANCE/METHOD/seed_k/elite_archive.json
```

## مشاهده وضعیت آموزش در Kaggle

دستور `run_ppo_suite` به طور پیش فرض در شروع و پایان هر روش و تقریباً پس از
هر ۲۵۶ اپیزود، وضعیت آموزش را بدون تأخیر در خروجی Notebook چاپ می‌کند. برای
تغییر فاصله گزارش‌ها از گزینه زیر استفاده کنید:

```text
--log-every-episodes 128
```

مثلاً برای اجرای هدف فعلی روی `D_14S_30P`:

```bash
python -m experiments.run_ppo_suite \
  --instances D_14S_30P \
  --profile full \
  --methods stable_mlp attention hierarchical hacipp hacipp_rl_improve \
  --log-every-episodes 128 \
  --device auto \
  --seed 42 \
  --output-directory results/ablation/D_14S_30P/seed_42
```

هر خط پیشرفت شامل شماره update، تعداد اپیزودهای انجام شده، میانگین و بهترین
Objective، بهترین جواب دیده شده، Lossهای PPO، Entropy، زمان سپری شده و ETA
است. چون PPO پس از جمع‌آوری یک batch کامل update می‌شود، گزارش در اولین مرز
update پس از فاصله درخواستی چاپ می‌شود. مقدار `0` چاپ دوره‌ای را غیرفعال
می‌کند:

```text
--log-every-episodes 0
```

ستون مهم:

```text
improvement_over_published_bfs_percent
```

- مثبت: روش ما از BFS منتشرشده Gurobi بهتر شده است؛
- صفر: برابر است؛
- منفی: هنوز از BFS پایین‌تر است.

## ۹. دانلود نتایج

```bash
%%bash
cd /kaggle/working/CIPP-RL-PPO-Suite
zip -qr /kaggle/working/CIPP_PPO_RESULTS.zip results
ls -lh /kaggle/working/CIPP_PPO_RESULTS.zip
```

بعد از بخش Output فایل ZIP را دانلود کنید یا Notebook را `Save Version` کنید.

## ۱۰. اجرای اختیاری Gurobi

فقط در صورتی که license معتبر در محیط Kaggle دارید:

```bash
python -m pip install -q -r requirements-gurobi.txt
```

سپس به دستور experiment اضافه کنید:

```text
--run-gurobi --gurobi-time-limit 3600
```

برای آزمایش hybrid جداگانه که بهترین جواب RL را به‌عنوان MIP start به Gurobi
می‌دهد، این flag را نیز اضافه کنید:

```text
--run-gurobi-warm-start
```

در leaderboard، standalone Gurobi و `rl_warm_start_gurobi_*` دو ردیف مستقل
دارند. این نسخه یک آزمایش hybrid است و به‌عنوان روش RL خالص گزارش نمی‌شود.

بدون این flag، جدول BFS و gap منتشرشده به‌صورت reference ثبت می‌شود. برای ادعای مقایسه‌ی هم‌زمان و هم‌سخت‌افزار، اجرای Gurobi روی همان Kaggle session لازم است.

## ۱۱. قرارداد قطعی ابعاد instance

هر instance تعداد locationها و دوره‌های خودش را تعیین می‌کند. در شناسه
`D_14S_30P`، مقدار `14S` یعنی ۱۴ location قابل‌ویزیت و `30P` یعنی ۳۰ دوره.
Idle یک action اضافه با اندیس صفر است؛ بنابراین این instance دقیقاً ۱۵ action
دارد:

```text
instance.n = 14 visit locations
instance.H = 30 periods
action 0 = Idle
actions 1..14 = visit locations
instance.num_actions = instance.n + 1 = 15
```

گزینه‌های مبهم `--total-states` و `--state-count-semantics` در pipeline اصلی
وجود ندارند. برای instance سفارشی نیز فایل JSON را بدهید؛ فیلدهای `n` و `H`
همان فایل بدون override شدن استفاده می‌شوند:

```bash
python -m experiments.run_ppo_suite \
  --instance-files data/instances/my_instance.json \
  --profile full \
  --methods hacipp hacipp_rl_improve \
  --device auto
```
