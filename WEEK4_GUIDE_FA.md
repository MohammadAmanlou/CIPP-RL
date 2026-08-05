# راهنمای کامل Week 4: PPO و Imitation Learning برای CIPP

## پاسخ دو سؤال اصلی

### PPO روی چه چیزی train می‌شود؟

PPO نباید روی `CIPP-D.xls` یا `CIPP-R.xls` آموزش ببیند، چون این دو
فایل benchmark واقعی و test نهایی هستند. اگر مدل روی همین داده‌ها
train شود، مقایسه با Gurobi اعتبار علمی ندارد.

در این پیاده‌سازی:

- train: تعداد ۱۲۸ instance مصنوعی با `n=14` و `H=30`؛
- validation: تعداد ۲۰ instance مصنوعی دیگر با seedهای ثابت و جدا؛
- test: دو instance واقعی `D_14S_30P` و `R_14S_30P` از Excel استاد؛
- در test وزن‌های مدل کاملاً freeze هستند و هیچ update یا retraining
  انجام نمی‌شود.

instanceهای مصنوعی از لحاظ مقیاس reward، cost، تابع زمانی U-shaped،
محدودیت استراحت، `q=12`، `w=2` و `alpha=7` شبیه مقاله‌اند، ولی
مقادیر reward و cost فایل واقعی را نمی‌خوانند. پس مدل ساختار مسئله را
یاد می‌گیرد، نه جواب benchmark را.

### آیا PPO معمولی rollout دارد؟

بله. PPO ذاتاً یک الگوریتم on-policy است و بدون rollout معنی ندارد.
در هر update، policy فعلی چند campaign کامل تولید می‌کند:

1. محیط reset می‌شود.
2. در هر روز policy یک توزیع categorical روی actionها می‌سازد.
3. action mask قبل از sample اعمال می‌شود.
4. action، reward، value، log-probability و mask ذخیره می‌شوند.
5. بعد از پایان campaign، با GAE advantage محاسبه می‌شود.
6. PPO با clipped objective چند epoch روی همان rollout تازه update
   می‌شود.

دو کاربرد rollout را نباید قاطی کرد:

- **Training rollout:** داده on-policy لازم برای update کردن PPO.
- **Evaluation rollout:** تولید چند مسیر با checkpoint freeze‌شده و
  انتخاب بهترین مسیر با objective دقیق CIPP.

در جدول نهایی هر دو حالت evaluation گزارش می‌شوند:

- `PPO-Greedy-1`: فقط یک مسیر، در هر state محتمل‌ترین action feasible.
- `PPO-Best-of-30`: سی مسیر stochastic feasible و انتخاب بهترین مسیر.

`Best-of-30` هیچ backtracking داخل یک episode ندارد. این روش ۳۰
campaign کامل مستقل می‌سازد و بهترین آن‌ها را برمی‌گرداند.

## ناسازگاری مهم مقاله و کد استاد

در فایل‌ها دو تعریف متفاوت وجود دارد:

| حالت | repeat penalty | budget | کاربرد |
|---|---:|---:|---|
| `paper_equation` | `mu_j = 1 - 0.04j` | فعال و متناسب با horizon | تعریف متن مقاله |
| `professor_code` | `mu_j = 1 - 0.04(j-1)` | در کد July 13 کامنت شده | مقایسه مستقیم با Table 5 |

نتیجه‌های این دو حالت نباید در یک ستون gap با هم مقایسه شوند. کد این
موضوع را enforce می‌کند و اگر checkpoint با variant دیگری train شده
باشد، benchmark خطا می‌دهد.

برای بازتولید جدول استاد، در تمام سه مرحله imitation، PPO و benchmark
از `--objective-variant professor_code` استفاده کنید. برای آزمایش
علمی مطابق معادله متن مقاله، هر سه مرحله را جداگانه با
`paper_equation` اجرا کنید.

## اجزای پیاده‌سازی

### PPO

فایل اصلی `src/models/ppo_agent.py` شامل موارد زیر است:

- Actor-Critic MLP؛
- policy head برای `n+1` action؛
- value head برای `V(s)`؛
- masked categorical distribution؛
- GAE و PPO clipped loss؛
- value loss، entropy bonus و gradient clipping؛
- checkpoint شامل model، optimizer، normalizer، config و metadata.

### Imitation Learning

فایل `src/models/imitation_model.py`:

1. itinerary به دنباله `(s_t, a_t*, mask_t)` تبدیل می‌شود.
2. policy با masked cross-entropy pre-train می‌شود.
3. checkpoint حاصل initialization مدل PPO دوم می‌شود.
4. همان policy با rolloutهای on-policy fine-tune می‌شود.

### حل دقیق

فایل `src/optimization/cipp_gurobi.py` از متغیرهای `Z`, `S`, `Y`,
`V` و تمام محدودیت‌های CIPP استفاده می‌کند. SciPy/HiGHS فقط fallback
برای تست محلی است و هیچ‌وقت با نام Gurobi گزارش نمی‌شود.

## اجرای مرحله‌به‌مرحله

تمام فرمان‌ها را از root پروژه اجرا کنید.

### نصب

در Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install gurobipy
```

در Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install gurobipy
```

قبل از ادامه مطمئن شوید ساخت یک `gurobipy.Model()` با license شما
کار می‌کند.

### تست کل پروژه

```bash
python -m pytest -q
```

خروجی تأییدشده این نسخه:

```text
51 passed
```

### آموزش PPO عادی

```bash
python -m experiments.train_ppo \
  --output-directory results/week4/ppo \
  --objective-variant professor_code \
  --updates 500 \
  --episodes-per-update 16 \
  --training-instances 128 \
  --validation-instances 20 \
  --normalizer-instances 32 \
  --validation-interval 10 \
  --number-of-states 14 \
  --horizon 30 \
  --hidden-dimension 256 \
  --learning-rate 0.0003 \
  --discount-factor 1.0 \
  --gae-lambda 0.95 \
  --clip-epsilon 0.2 \
  --entropy-coefficient 0.01 \
  --update-epochs 10 \
  --minibatch-size 256 \
  --seed 42 \
  --device auto
```

تقریباً `500 × 16 × 30 = 240,000` transition جمع می‌شود.

خروجی‌ها:

```text
results/week4/ppo/
  checkpoint_best.pt
  checkpoint_last.pt
  experiment_config.json
  training_history.json
  learning_curve.png
```

### ساخت imitation initialization با Gurobi

```bash
python -m experiments.train_imitation \
  --output-directory results/week4/imitation \
  --objective-variant professor_code \
  --solver gurobi \
  --teacher-instances 8 \
  --teacher-time-limit 300 \
  --epochs 100 \
  --batch-size 256 \
  --learning-rate 0.0003 \
  --hidden-dimension 256 \
  --seed 42 \
  --device cpu
```

به‌صورت پیش‌فرض اگر جواب teacher proven optimal نباشد، برنامه متوقف
می‌شود. برای نتیجه اصلی از `--allow-nonoptimal-teachers` استفاده
نکنید.

### PPO با imitation initialization

```bash
python -m experiments.train_ppo \
  --output-directory results/week4/imitation_ppo \
  --objective-variant professor_code \
  --initial-checkpoint \
    results/week4/imitation/imitation_initialization.pt \
  --updates 500 \
  --episodes-per-update 16 \
  --training-instances 128 \
  --validation-instances 20 \
  --validation-interval 10 \
  --seed 42 \
  --device auto
```

### جدول نهایی روی کوچک‌ترین instanceهای واقعی

```bash
python -m experiments.benchmark_week4 \
  --ppo-checkpoint results/week4/ppo/checkpoint_best.pt \
  --imitation-ppo-checkpoint \
    results/week4/imitation_ppo/checkpoint_best.pt \
  --dqn-checkpoint results/week3/dqn/checkpoint_best.pt \
  --democrat-excel CIPP-D.xls \
  --republican-excel CIPP-R.xls \
  --parties both \
  --number-of-states 14 \
  --horizon 30 \
  --objective-variant professor_code \
  --rollouts 30 \
  --output-directory results/week4/benchmark \
  --seed 42 \
  --device cpu
```

اگر DQN checkpoint ندارید، آرگومان `--dqn-checkpoint` را حذف کنید.
ZIP ارسالی کلاس DQN دارد، ولی checkpoint آموزش‌دیده DQN در آن نیست.

برای اجرای تازه Gurobi و تولید log، LP و solution:

```bash
--run-exact --exact-solver gurobi --exact-time-limit 3600
```

بدون `--run-exact`، ردیف Gurobi از Table 5 مقاله خوانده می‌شود.

## خروجی جدول

```text
results/week4/benchmark/
  comparison_table.csv
  comparison_table.md
  comparison_table.tex
  benchmark_details.json
```

ستون‌ها شامل objective، gap، CPU time، feasibility، تعداد violation،
تعداد rollout، budget utilization، idle days، unique states و HHI
هستند. فایل JSON itinerary عددی و نام stateهای هر مسیر را هم دارد.

## پروتکل نتیجه علمی نهایی

هر آزمایش را حداقل با seedهای `42`, `43`, `44` اجرا و میانگین و انحراف
معیار را گزارش کنید. انتخاب checkpoint فقط بر اساس validation مصنوعی
ثابت انجام می‌شود. نتیجه test واقعی نباید برای انتخاب hyperparameter،
تعداد update یا checkpoint استفاده شود.

فولدر `results/week4/smoke` فقط نشان می‌دهد pipeline اجرا می‌شود. مدل
آن تنها دو PPO update و teacher غیرنهایی داشته و نباید در مقاله یا
گزارش به‌عنوان performance نهایی استفاده شود.
