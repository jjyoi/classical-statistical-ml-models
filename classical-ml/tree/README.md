## Quick Notes
#### LGBMRegressor 
# LightGBM Regression — Quick Review

Predict `daily_kwh` from floor area, outdoor temperature, occupants, and heating type. **MAE** is the average absolute error in kWh. Lower = better.

### Pandas and NumPy
| Syntax | What it does |
|---|---|
| `df.dtypes` / `type(s.iloc[0])` | Check column types / check one value’s type. `np.str_` is NumPy’s string type. |
| `s.isna().sum()` / `s.value_counts(dropna=False)` | Count missing values / inspect labels and their counts. `"unknown"` is text until converted. |
| `s.str.strip().str.lower()` | Clean a whole text column. For one string, use `value.strip().lower()`. Use `.apply()` for custom logic. |
| `pd.to_numeric(s, errors="coerce")` | Convert text to numbers. Invalid text becomes `NaN`. Check failed conversions and invalid ranges. |
| `s.astype("category")` | Mark a feature like `heating_type` as categorical. LightGBM also handles numeric `NaN`s. |
| `df.loc[s.eq("electric")].copy()` | Filter rows and make a separate table to edit. `.eq()` compares the whole column. |
| `np.where(condition, x, y)` | Pick `x` if true, else `y`, for each row. With `< median`, ties go to `y`. |
| `(temp // 5) * 5` | Get the start of a 5°C band. `12.7 → 10`, `-2 → -5`. |
| `groupby([...])[col].median().unstack("size")` | Find each group’s median, then turn size groups into columns. `.plot()` uses the index as x and draws one line per column. |
| `X.assign(new_col=values)` | Return a DataFrame with the new column. Apply the same formula to train and test. Check `X.columns` before fitting. |

### feature engineering
- The plot showed a larger gap between big and small electric homes on colder days. That suggested an **area × temperature interaction**.
- First try `area_m2 * outdoor_temp_c`. Then try `area_m2 * (20 - outdoor_temp_c).clip(lower=0)`.
- `.clip(lower=0)` is the ReLU step. It sets negative values to zero. **20°C was a cutoff to test**, chosen from the plot. Keep both original features when testing the product.
- In Python, `(20 - s, 0)` makes a tuple. Use `(20 - s).clip(lower=0)`.

### model and tuning

- **Baseline:** Predict the training median for every row. **`LGBMRegressor`** predicts a number by adding adjustments from many trees.
- **`num_leaves`** caps leaves per tree. **`min_child_samples`** controls how small a leaf can be. **`max_depth`** caps branch depth.
- **`learning_rate`** controls the size of each tree’s adjustment. **`n_estimators`** controls how many trees to build. Try smaller steps with more trees.
- **`GridSearchCV`** tries settings on training folds and refits the winner on all training rows by default. `best_params_` gives the settings. `best_estimator_` gives the fitted model. `cv_results_` gives fold scores. Negate `neg_mean_absolute_error` scores to get MAE.
- Set `random_state` on the train/test split and use the same CV folds for comparisons. Compare feature versions with the same model settings to isolate the feature’s effect. Use CV to choose and the test set for the final check.

### what happened
- Tuning lowered test MAE from **3.821 → 3.401 kWh**.
- The engineered feature lowered test MAE to **3.396 kWh**. Mean CV MAE went **3.736 → 3.720**, with improvement in **3 of 5 folds**. I’d keep the tuned model with the original features because the gain was tiny and mixed.

**Offline debugging:** `type(obj)` → `dir(obj)` → `help(obj.method)`, e.g. `help(pd.DataFrame.assign)`.