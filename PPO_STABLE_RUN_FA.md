# اجرای پایدار و سریع‌تر PPO برای گزارش Week 4

## نصب Patch روی پروژه فعلی

فایل `CIPPRL_PPO_Stable_v2_Patch.zip` را کنار پوشه فعلی پروژه قرار دهید و در ریشه همان پروژه فعلی اجرا کنید:

```cmd
cd /d C:\Users\Lenovo\Desktop\CIPPRL_Week4_PPO_Complete\CIPPRL
powershell -NoProfile -Command "Expand-Archive -Path '..\CIPPRL_PPO_Stable_v2_Patch.zip' -DestinationPath '.' -Force"
.venv\Scripts\activate
python -m pytest -q
```

Patch فقط فایل‌های PPO و راهنمای جدید را جایگزین می‌کند و به checkpointها و نتایج قبلی دست نمی‌زند.

## چرا اجرای قبلی قابل استفاده نبود؟

در اجرای قبلی با seed 42:

- feasibility در تمام اپیزودها برابر `1.000` بود؛ بنابراین action masking و قیود درست کار می‌کردند.
- بهترین validation در update 120 برابر `9622.283` بود.
- میانگین Greedy روی دقیقاً همان ۲۰ validation instance حدود `17191.873` بود.
- میانگین Random تک‌مسیره روی همان توزیع حدود `11792` بود.
- میانگین rollout آموزشی از `11433.936` در updateهای 1 تا 50 به `9702.268` در updateهای 301 تا 350 کاهش یافت.

بنابراین PPO حتی از Random نیز ضعیف‌تر شده بود. علت اصلی این بود که critic با reward و return خام در مقیاس حدود ده‌هزار آموزش می‌دید، درحالی‌که advantage مربوط به actor نرمال می‌شد. به‌دلیل backbone مشترک، value loss بسیار بزرگ جهت گرادیان را غالب می‌کرد. همچنین ۱۰ epoch روی تنها ۴۸۰ transition هر update، هم زمان اجرا را بالا می‌برد و هم خطر over-update را زیاد می‌کرد.

در نسخه جدید:

- فقط reward مورد استفاده در GAE و critic با `reward_scale=0.001` مقیاس می‌شود.
- objective واقعی CIPP در validation، benchmark و گزارش هیچ تغییری نمی‌کند.
- Random و Greedy روی validation پیش از train چاپ می‌شوند.
- فاصله PPO تا Greedy در هر validation چاپ می‌شود.
- early stopping و تاریخچه موقت اضافه شده است.

## اجرای کالیبراسیون seed 42

اگر Patch را روی پروژه فعلی نصب کرده‌اید، در همان CMD ویندوز:

```cmd
cd /d C:\Users\Lenovo\Desktop\CIPPRL_Week4_PPO_Complete\CIPPRL
.venv\Scripts\activate
python -m pytest -q
```

سپس:

```cmd
set SEED=42

python -m experiments.train_ppo ^
--output-directory results/week4/report_runs_v2/ppo_seed%SEED% ^
--objective-variant professor_code ^
--updates 200 ^
--episodes-per-update 24 ^
--training-instances 128 ^
--validation-instances 20 ^
--normalizer-instances 32 ^
--validation-interval 10 ^
--number-of-states 14 ^
--horizon 30 ^
--hidden-dimension 128 ^
--learning-rate 0.0002 ^
--reward-scale 0.001 ^
--discount-factor 1.0 ^
--gae-lambda 0.95 ^
--clip-epsilon 0.2 ^
--value-loss-coefficient 0.5 ^
--entropy-coefficient 0.005 ^
--gradient-clip-norm 0.5 ^
--update-epochs 4 ^
--minibatch-size 256 ^
--early-stopping-patience 6 ^
--early-stopping-min-delta 1.0 ^
--minimum-updates 80 ^
--seed %SEED% ^
--device auto
```

این تنظیم حداکثر `200 × 24 × 30 = 144000` transition می‌سازد و هر batch را فقط چهار بار استفاده می‌کند. در مقایسه با اجرای قبلی، تعداد transitionها ۴۰ درصد کمتر و تعداد پردازش نمونه‌ها در مرحله optimizer حدود ۷۶ درصد کمتر است.

## معیار پذیرش قبل از اجرای سه seed

ابتدای اجرا دو مرجع چاپ می‌شوند:

```text
validation_references random_one_mean=... greedy_mean=...
```

در هر ۱۰ update نیز این مقدار چاپ می‌شود:

```text
validation_mean=... gap_to_greedy=...%
```

فقط در صورتی seedهای 43 و 44 را اجرا کنید که checkpoint منتخب seed 42:

1. از `random_one_mean` بهتر باشد؛
2. روند validation صعودی یا دست‌کم پایدار داشته باشد؛
3. feasibility برابر `1.000` باقی بماند؛
4. ترجیحاً فاصله آن تا Greedy کمتر از ۲۰ درصد شود.

اگر تا update 80 هنوز از Random پایین‌تر بود، سه seed را ادامه ندهید و ابتدا diagnostics را بررسی کنید.

## بررسی checkpoint منتخب

```cmd
python -c "import torch; p=torch.load(r'results/week4/report_runs_v2/ppo_seed42/checkpoint_best.pt', map_location='cpu', weights_only=False); m=p['training_metadata']; print('best update:',m['selected_update']); print(m['best_validation']); print(m['validation_baselines'])"
```

## اجرای seedهای دیگر

پس از موفقیت seed 42، همان فرمان را با این دو مقدار تکرار کنید:

```cmd
set SEED=43
```

و سپس:

```cmd
set SEED=44
```

پوشه خروجی به‌کمک `%SEED%` خودکار جدا می‌شود.

## Imitation + PPO

فرمان ساخت teacherهای Gurobi تغییری نکرده است. برای fine-tuning از checkpoint imitation، همان تنظیمات پایدار بالا را استفاده کنید و فقط این دو گزینه را اضافه/تغییر دهید:

```cmd
--output-directory results/week4/report_runs_v2/imitation_ppo_seed%SEED% ^
--initial-checkpoint results/week4/report_runs_v2/imitation_seed%SEED%/imitation_initialization.pt ^
```

از `checkpoint_best.pt` برای benchmark نهایی استفاده کنید، نه `checkpoint_last.pt`.
