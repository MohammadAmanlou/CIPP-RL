# روش‌ها و Ablationهای پیاده‌سازی‌شده

## روش‌های construction

| نام CLI | معماری | هدف مقایسه |
|---|---|---|
| `stable_mlp` | Actor/Critic جدا، PPO پایدار، Huber و value clipping | کنترل PPO قوی‌تر از کد قدیمی |
| `attention` | token هر location و self-attention | اثر نمایش ساختارمند و permutation equivariance |
| `hierarchical` | Attention و تصمیم سلسله‌مراتبی Idle/Visit/location | اثر Idle planner |
| `hacipp` | Hierarchical Attention + marginal residual + count planner + POMO/group-relative + elite replay | مدل construction اصلی |
| `hacipp_rl_improve` | HACIPP construction + improvement policy آموزش‌دیده با PPO | روش اصلی نهایی |

## تغییرهای PPO

- Actor و Critic کاملاً جدا در معماری‌های attention؛
- `discount=1` و `GAE lambda=1` برای افق کوتاه deterministic؛
- reward scale تطبیقی براساس scale همان instance؛
- Huber value loss و clipped value target؛
- target-KL early stopping؛
- entropy نرمال‌شده براساس تعداد actionهای مجاز؛
- entropy و learning-rate annealing؛
- auxiliary Q head برای تخمین return هر action بدون استفاده‌ی نادرست از Q به‌عنوان baseline policy gradient؛
- count planner با targetهای self-generated؛
- archive بهترین جواب‌ها و self-imitation فقط از rolloutهای خود عامل؛
- POMO-style شروع‌های متنوع و group-relative advantage.

## جست‌وجوی frozen policy

- deterministic decoding؛
- Best-of-K stochastic RL rollouts؛
- simulation-guided beam search که prefixها را فقط با policy آموزش‌دیده کامل می‌کند؛
- هیچ Greedy یا Gurobi در SGBS استفاده نمی‌شود.

## RL Improvement Policy

عامل دوم از جواب HACIPP شروع می‌کند و در هر مرحله از میان neighborhoodهای feasible یکی را انتخاب می‌کند:

- Replace location؛
- Swap two periods؛
- Idle relocation از طریق Swap؛
- rolling-window rotation؛
- rolling-window shuffle؛
- Stop.

Candidateها به‌صورت vectorized با همان تابع هدف و تمام constraintها بررسی می‌شوند. پاداش عامل دقیقاً تغییر objective است. عامل می‌تواند برای خروج از local optimum حرکت موقتاً ضعیف‌تر انجام دهد، ولی بهترین incumbent جدا نگه داشته می‌شود؛ بنابراین جواب نهایی هیچ‌گاه از starting solution بدتر نیست.

## جدول ablation مقاله

ترتیب پیشنهادی گزارش:

```text
Greedy
Stable-MLP-PPO
Attention-PPO
Hierarchical-Attention-PPO
HACIPP construction greedy
HACIPP Best-of-K
HACIPP SGBS
HACIPP + RL Improvement-30
Gurobi BFS / bound / gap
```

برای هر روش objective، runtime، feasibility، unique locations، HHI، gap نسبت به BFS و در instanceهای optimal فاصله با optimum را گزارش کنید.

