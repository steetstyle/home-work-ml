"""Generate code_soyad.ipynb from template."""
import json
import uuid
from pathlib import Path
from textwrap import dedent


def md(s: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": str(uuid.uuid4()),
        "source": dedent(s).strip().splitlines(True),
    }


def code(s: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "id": str(uuid.uuid4()),
        "source": dedent(s).strip().splitlines(True),
    }


def main():
    cells = []

    cells.append(
        md(
            """
            # Earnings Sonrası Hisse Getiri Tahmini (Regression)

            **Dosya adı:** `code_soyad.ipynb` — `STUDENT_SURNAME = "soyad"` değerini kendi soyadınızla değiştirin.

            **Tanım (sızıntı önleme):** `event_date` = yfinance earnings tarihi. **Giriş günü** (`entry_date`) = bu tarihten *sonraki* ilk iş günü.
            **Hedefler** (`y_*`): giriş günü kapanışından itibaren 1, 3, 5, 10, 15, 21, 63 iş günü ilerideki kapanışa kadar basit getiri.
            **Ticker listesi:** [tickers.csv](tickers.csv) (`sector`, `ticker` sütunları). Dosya yoksa varsayılan evren kullanılır ve aynı dizinde `tickers.csv` oluşturulur; düzenledikten sonra `from odev_helpers import reload_sector_tickers; reload_sector_tickers()` ile yenileyin (kernel yeniden başlatmak da yeterli).

            **Özellikler:** `feature_date` = girişten bir iş günü önce. Piyasa/sektör sinyalleri **ETF yok**: tüm evrenin ve (kendi hariç) sektör arkadaşlarının eşit ağırlıklı kapanış sepeti. EPS yfinance’dan; eksikler NaN kalabilir.

            **Bu not defterinde:** Aşağıdaki bölümlerde her önemli tablo ve grafik için kısa bir **Okuma rehberi** (ne işe yarar / nasıl okunur) vardır; raporda cümlelerinizi bu başlıklardan türetebilirsiniz.
            """
        )
    )

    cells.append(
        code(
            """
            import importlib
            import warnings
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns

            warnings.filterwarnings("ignore")

            ROOT = Path.cwd()
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            STUDENT_SURNAME = "soyad"

            # Jupyter aynı kernelde eski odev_*.py önbelleğini tutar; hücreyi tekrar çalıştırınca güncel kodu yükle
            for _m in ("odev_helpers", "odev_modeling", "odev_export"):
                if _m in sys.modules:
                    importlib.reload(sys.modules[_m])

            from odev_helpers import (
                CORR_WINDOW_TITLE_TR,
                EARNINGS_CACHE_CSV,
                RAW_PRICES_LONG_CSV,
                assemble_dataset,
                correlation_bundles_by_calendar_windows,
                drop_high_corr_columns,
                feature_columns,
                primary_corr_from_window_bundles,
                slice_returns_calendar_tail,
                target_columns,
            )
            from odev_modeling import (
                add_sector_peer_returns,
                add_ticker_dummies,
                correlation_prune_cv,
                cv_eval,
                grid_regularized,
                linear_baseline_cv,
                numeric_pipeline,
                randomized_lgbm,
                randomized_xgb,
                rfe_cv,
                select_k_best_cv,
                time_series_cv,
                prepare_xy,
            )
            from sklearn.linear_model import Ridge, Lasso
            from sklearn.model_selection import cross_val_score, learning_curve

            from odev_export import export_excel, export_presentation

            sns.set_theme(style="whitegrid", context="talk")
            plt.rcParams["figure.figsize"] = (10, 5)
            """
        )
    )

    cells.append(md("## 1. Veri çekme ve panel"))
    cells.append(
        code(
            """
            # Ham fiyatlar data/raw_prices_long.csv önbelleğe yazılır; varsa indirme atlanır.
            events, adj, rets = assemble_dataset(progress=True)
            print("raw cache:", RAW_PRICES_LONG_CSV)
            print("earnings cache:", EARNINGS_CACHE_CSV)
            print("events", events.shape, "adj", adj.shape)
            events.head()
            """
        )
    )

    cells.append(
        md(
            """
            ### Neden: Veri
            - **Neden böyle ilerledik:** Tüm hisseler tek indirmede hizalanır; ham panel `data/raw_prices_long.csv` olarak saklanır, dosya varsa ve tüm ticker’lar kapsanıyorsa tekrar indirilmez (`force_refresh_raw=True` veya dosyayı silerek yenilersiniz). **Kazanç tarihleri** ticker başına yfinance’tan yavaş gelir; `data/earnings_dates_cache.csv` + `earnings_cache_meta.json` ile aynı ticker listesi ve limit için ikinci çalıştırmadan itibaren tekrar çağrılmaz (`force_refresh_earnings=True` veya bu iki dosyayı silerek yenileyin).
            - **Neden bunu seçtik:** Tekrarlanabilir, ücretsiz `yfinance` kaynağı; tekrar çalıştırmada hız ve kota tasarrufu.
            - **Alternatif:** Finnhub / Polygon — API anahtarı gerekir, bu ödevde kullanılmadı.
            - **Sonraki adım:** Eksik EPS için manuel CSV birleştirmek.

            ### Okuma rehberi — §1 Çıktılar
            **`events.shape` / `adj.shape`**
            - **Ne işe yarar:** Veri boyutu; model için yeterli satır var mı, kaç ticker kapsandı hızlı kontrol.
            - **Nasıl okunur:** `(satır, sütun)`; satır = olay sayısı, `adj` sütunları = indirilen hisse sayısı. Satır çok düşükse CV güvenilmez.

            **`events.head()`**
            - **Ne işe yarar:** Tarih, ticker, birkaç özellik ve hedef sütunlarının gerçekten hizalandığını görmek.
            - **Nasıl okunur:** `entry_date` ≤ `feature_date` ilişkisine dikkat (özellikler girişten önceki güne ait). `y_*` çok uç değerler aykırı olay olabilir.

            **İlk çalıştırma uzun sürerse**
            - Ham fiyat ve kazanç önbelleği oluşana kadar ağ istekleri yapılır; sonraki çalıştırmalar `data/` altındaki CSV/JSON dosyalarından okur ve genelde çok daha kısadır. `tickers.csv` değiştiyse veya daha fazla kazanç geçmişi istiyorsanız (`earnings_limit`) ilgili önbellek dosyalarını silin veya `force_refresh_*` bayraklarını kullanın.
            """
        )
    )

    cells.append(md("## 2. Korelasyon analizi"))
    cells.append(
        code(
            """
            from IPython.display import display

            univ = [c for c in adj.columns]
            ret_w = adj[univ].pct_change()

            corr_bundles = correlation_bundles_by_calendar_windows(ret_w)

            win_tab = pd.DataFrame(
                [
                    {
                        "pencere": CORR_WINDOW_TITLE_TR.get(k, k),
                        "ay": v["months"],
                        "n_gun": v["n_obs"],
                    }
                    for k, v in corr_bundles.items()
                ]
            )
            display(win_tab)

            for k, meta in corr_bundles.items():
                cmat = meta["corr"]
                ttl = CORR_WINDOW_TITLE_TR.get(k, k) + f" (n={meta['n_obs']} gün)"
                if cmat.empty or cmat.shape[0] < 2:
                    print(ttl, "→ yeterli gözlem yok, ısı haritası atlandı")
                    continue
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(cmat, ax=ax, cmap="vlag", center=0, square=True)
                ax.set_title(f"Getiri korelasyon — {ttl}")
                plt.tight_layout()
                plt.show()

            for k in ("1y", "2y"):
                meta = corr_bundles.get(k)
                if not meta or meta["corr"].empty or meta["corr"].shape[0] < 2:
                    continue
                g = sns.clustermap(
                    meta["corr"], cmap="vlag", center=0, figsize=(12, 12)
                )
                g.fig.suptitle(
                    f"Clustermap — {CORR_WINDOW_TITLE_TR.get(k, k)}", y=1.02
                )
                plt.show()

            if "NVDA" in ret_w.columns and "INTC" in ret_w.columns:
                plt.figure(figsize=(11, 5))
                ax = plt.gca()
                for k, meta in corr_bundles.items():
                    sub = slice_returns_calendar_tail(ret_w, meta["months"])
                    if len(sub) < 15:
                        continue
                    w = min(60, max(10, len(sub) // 3))
                    if len(sub) < w + 2:
                        continue
                    r = sub["NVDA"].rolling(w).corr(sub["INTC"]).dropna()
                    ax.plot(r.index, r.values, lw=1.2, label=CORR_WINDOW_TITLE_TR.get(k, k))
                ax.set_title("NVDA vs INTC — rolling korelasyon (her pencerede farklı w)")
                ax.legend(loc="best", fontsize=8)
                plt.tight_layout()
                plt.show()
            """
        )
    )

    cells.append(
        md(
            """
            ### Neden: Korelasyon
            - **Neden:** Multi-stock model için ortak hareket var mı diye bakıldı; **son 1–24 ay** takvim pencerelerinde Pearson korelasyonu (daha kısa örneklerde rejim / gürültü farkı).
            - **Seçim:** Her pencere için ısı haritası; 1 yıl / 2 yıl için clustermap; NVDA–INTC rolling örnek (pencere içi veri).
            - **Alternatif:** DCC-GARCH — kapsam dışı.

            ### Okuma rehberi — §2 Grafikler ve tablo
            **Pencere özeti tablosu (`pencere`, `ay`, `n_gün`)**
            - **Ne işe yarar:** Her korelasyon matrisinde kaç günlük getiri kullanıldığını gösterir; kısa pencerelerde korelasyonun daha gürültülü olabileceğini hatırlatır.
            - **Nasıl okunur:** Aynı `ay` için `n_gün` beklediğiniz iş günü sayısıyla uyumlu mu bakın; çok düşükse o pencerenin ısı haritasını zayıf güvenle yorumlayın.

            **Isı haritası (her pencere)**
            - **Ne işe yarar:** Hangi hisselerin getirilerinin aynı yönde hareket ettiğini (Pearson ρ) gösterir; çoklu collinearity riski için.
            - **Nasıl okunur:** Hücre rengi: koyu kırmızı = güçlü pozitif, mavi = negatif, beyaza yakın = zayıf ilişki. Köşegen her zaman 1 (kendiyle). Sektör kümesi blokları görüyorsanız “ortak faktör” vardır.

            **Clustermap (1y, 2y)**
            - **Ne işe yarar:** Benzer korelasyon profiline sahip ticker’ları ağaç yapısıyla gruplar.
            - **Nasıl okunur:** Dendrogramda erken birleşen dallar birbirine yakın hareket eden hisseler; raporda “küme” diye özetlenebilir.

            **NVDA–INTC çok çizgili rolling korelasyon**
            - **Ne işe yarar:** Aynı çift için farklı tarih dilimlerinde ilişkinin zamanla nasıl değiştiğini kıyaslamak.
            - **Nasıl okunur:** Çizgiler üst üste binmez; etiket hangi pencereye ait. Yüksek eğri = o dönemde güçlü pozitif korelasyon. Her pencerede `w` (rolling uzunluğu) farklı olduğu için mutlak seviyeyi pencereler arası kıyaslamada dikkatli olun; trend ve göreli sıralama daha güvenli.
            """
        )
    )

    cells.append(md("## 3. Özellikler ve hedef dağılımları"))
    cells.append(
        code(
            """
            feats = feature_columns(events)
            print("Özellik sayısı:", len(feats))

            for tcol in target_columns():
                plt.figure(figsize=(4, 3))
                events[tcol].dropna().hist(bins=40)
                plt.title(f"Hedef {tcol}")
                plt.tight_layout()
                plt.show()
            """
        )
    )

    cells.append(
        md(
            """
            ### Okuma rehberi — §3 Özellik sayısı ve hedef histogramları
            **`Özellik sayısı`**
            - **Ne işe yarar:** Modele giren sayısal sütun adedi; aşırı çok özellik → overfit / yavaş eğitim riski.
            - **Nasıl okunur:** Sayı makul mü (ör. olay sayısının çok üstünde değil mi). Sonraki bölümlerde seçim (RFE, KBest, corr prune) bu set üzerinde.

            **Her `y_*` için histogram**
            - **Ne işe yarar:** Hedef değişkenin dağılımı (çarpıklık, aykırı, sıfıra yakın kümeleşme); RMSE yorumu ve model seçimi için.
            - **Nasıl okunur:** Tek tepeli ve simetrike yakınsa linear modeller daha makul; uzun kuyruk veya uç değer çokluğu → hata metrikleri birkaç aykırı olaya duyarlı olur, robust/ensemble düşünülebilir. Uzun horizon (`y_*` büyük) genelde daha geniş yayılım gösterir.
            """
        )
    )

    cells.append(md("## 4. Linear pipeline — her hedef için en iyi FS + reg"))
    cells.append(
        code(
            """
            cv = time_series_cv(5)
            per_target = {}


            def _fs_feature_matrix(fs_name, X, y, bundles):
                if fs_name == "baseline":
                    return X.copy()
                if fs_name == "corr_prune":
                    return bundles["corr_prune"]["X"].copy()
                pipe = bundles[fs_name]["model"]
                pipe.fit(X, y)
                if fs_name == "kbest":
                    return X.loc[:, pipe.named_steps["kb"].get_support()].copy()
                if fs_name == "rfe":
                    return X.loc[:, pipe.named_steps["rfe"].support_].copy()
                return X.copy()


            for tgt in target_columns():
                X_all, y_all, _ = prepare_xy(events, tgt, feats)
                bundles = {}
                bundles["baseline"] = linear_baseline_cv(X_all, y_all, cv)
                bundles["corr_prune"] = correlation_prune_cv(X_all, y_all, cv)
                bundles["kbest"] = select_k_best_cv(X_all, y_all, cv)
                bundles["rfe"] = rfe_cv(X_all, y_all, cv)
                tab_fs = (
                    pd.DataFrame(
                        {
                            "yöntem": list(bundles.keys()),
                            "cv_rmse_mean": [bundles[k]["cv_rmse_mean"] for k in bundles],
                            "cv_rmse_std": [bundles[k]["cv_rmse_std"] for k in bundles],
                        }
                    )
                    .set_index("yöntem")
                )
                s = tab_fs["cv_rmse_mean"].replace([np.inf, -np.inf], np.nan)
                if s.dropna().empty:
                    continue
                best_fs = str(s.idxmin())
                best_X = _fs_feature_matrix(best_fs, X_all, y_all, bundles)
                res_ridge = grid_regularized(best_X, y_all, cv, kind="ridge")
                res_lasso = grid_regularized(best_X, y_all, cv, kind="lasso")
                res_enet = grid_regularized(best_X, y_all, cv, kind="elastic")
                lin_rmse = {
                    "Ridge": float(res_ridge["cv_rmse_mean"]),
                    "Lasso": float(res_lasso["cv_rmse_mean"]),
                    "ElasticNet": float(res_enet["cv_rmse_mean"]),
                }
                best_reg = min(lin_rmse, key=lin_rmse.get)
                cv_best_lin = min(lin_rmse.values())
                per_target[tgt] = {
                    "X": best_X,
                    "y": y_all,
                    "best_fs": best_fs,
                    "best_reg": best_reg,
                    "tab_fs": tab_fs.reset_index(),
                    "bundles": bundles,
                    "res_ridge": res_ridge,
                    "res_lasso": res_lasso,
                    "res_enet": res_enet,
                    "cv_best_lin": cv_best_lin,
                }

            tab_fs_all = pd.concat(
                [per_target[t]["tab_fs"].assign(hedef=t) for t in per_target],
                ignore_index=True,
            )
            display(tab_fs_all)

            summary_linear = pd.DataFrame(
                [
                    {
                        "hedef": t,
                        "best_fs": per_target[t]["best_fs"],
                        "best_reg": per_target[t]["best_reg"],
                        "cv_rmse_linear": per_target[t]["cv_best_lin"],
                    }
                    for t in per_target
                ]
            )
            display(summary_linear)

            viz_target = min(per_target, key=lambda t: per_target[t]["cv_best_lin"])
            print("Görselleştirme (path / learning / residual / SHAP):", viz_target)
            best_X = per_target[viz_target]["X"]
            y_all = per_target[viz_target]["y"]
            res_ridge = per_target[viz_target]["res_ridge"]
            res_lasso = per_target[viz_target]["res_lasso"]
            res_enet = per_target[viz_target]["res_enet"]
            tab_linear_fs = per_target[viz_target]["tab_fs"]

            reg_compare = pd.DataFrame(
                {
                    "model": ["Ridge", "Lasso", "ElasticNet"],
                    "cv_rmse_mean": [
                        res_ridge["cv_rmse_mean"],
                        res_lasso["cv_rmse_mean"],
                        res_enet["cv_rmse_mean"],
                    ],
                }
            )
            display(reg_compare)
            """
        )
    )

    cells.append(
        md(
            """
            ### Okuma rehberi — §4 Tablolar (her hedef)
            **`tab_fs_all`**
            - **Ne işe yarar:** Her `y_*` hedefi için baseline / corr_prune / kbest / rfe CV RMSE karşılaştırması.
            - **Nasıl okunur:** Satırlar `hedef` + `yöntem` ile gruplanır; her hedefte **en düşük** `cv_rmse_mean` kazanan FS olur (`best_fs`).

            **`summary_linear`**
            - **Ne işe yarar:** Her hedef için seçilen FS, ardından Ridge/Lasso/ENet arasından en iyi reg ve birleşik linear CV RMSE.
            - **Nasıl okunur:** `best_reg` o hedef için kullanılacak düzenlileştirilmiş model; `viz_target` satırı tüm hedefler içinde en düşük hatayı veren hedef (aşağıdaki grafikler onun üzerinden).

            **`reg_compare` (yalnızca `viz_target`)**
            - **Ne işe yarar:** Seçilen özellik matrisinde üç reglinin kıyası (görselleştirilen hedef).
            - **Nasıl okunur:** Düşük `cv_rmse_mean` iyi; bu tablo `viz_target` için özet.
            """
        )
    )

    cells.append(
        code(
            """
            from sklearn.linear_model import Ridge, Lasso

            dfp = best_X.copy()
            dfp["_y"] = y_all
            dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna(how="any")
            if len(dfp) < 5:
                print("Ridge/Lasso path: yetersiz tam satır, atlandı")
            else:
                Xmat = dfp.drop(columns=["_y"]).values.astype(float)
                ymat = dfp["_y"].values.astype(float)

                alphas = np.logspace(-3, 2, 25)
                coefs_r = []
                for a in alphas:
                    coefs_r.append(Ridge(alpha=a).fit(Xmat, ymat).coef_)
                coefs_r = np.vstack(coefs_r)
                plt.figure()
                plt.plot(alphas, coefs_r)
                plt.xscale("log")
                plt.title(f"Ridge katsayıları vs alpha — örnek hedef: {viz_target}")
                plt.xlabel("alpha")
                plt.show()

                coefs_l = []
                for a in alphas:
                    coefs_l.append(
                        Lasso(alpha=a, max_iter=10000).fit(Xmat, ymat).coef_
                    )
                coefs_l = np.vstack(coefs_l)
                plt.figure()
                plt.plot(alphas, coefs_l)
                plt.xscale("log")
                plt.title(f"Lasso katsayıları vs alpha — örnek hedef: {viz_target}")
                plt.xlabel("alpha")
                plt.show()
            """
        )
    )

    cells.append(
        md(
            """
            ### Okuma rehberi — §4 Ridge / Lasso katsayı yolları
            - **Ne işe yarar:** Ceza parametresi `alpha` arttıkça katsayıların küçülmesini (Ridge) veya sıfırlanmasını (Lasso) görmek; hangi özelliklerin “dayanıklı” kaldığını anlamak.
            - **Nasıl okunur:** Log-x ekseninde solda düşük ceza, sağda güçlü ceza. Lasso’da çizgilerin eksene düşmesi = o özellik eleniyor. **Uyarı:** Burada `StandardScaler` dışında ham X kullanıldı; katsayı büyüklükleri doğrudan “önem sırası” değildir; yön ve stabilite için kullanın.
            """
        )
    )

    cells.append(
        code(
            """
            train_sizes, train_scores, test_scores = learning_curve(
                res_ridge["model"],
                best_X,
                y_all,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                train_sizes=np.linspace(0.2, 1.0, 5),
                n_jobs=1,
            )
            train_rmse = -train_scores
            test_rmse = -test_scores
            plt.figure()
            plt.plot(train_sizes, train_rmse.mean(axis=1), label="train")
            plt.plot(train_sizes, test_rmse.mean(axis=1), label="cv")
            plt.legend()
            plt.title(f"Learning curve (Ridge) — örnek hedef: {viz_target}")
            plt.show()
            """
        )
    )

    cells.append(
        md(
            """
            ### Okuma rehberi — §4 Learning curve
            - **Ne işe yarar:** Eğitim verisi büyüdükçe train / CV hatasının nasıl değiştiğini gösterir; daha fazla veri fayda eder mi (underfitting) yoksa model zaten doydu mu.
            - **Nasıl okunur:** İki eğri birbirine yaklaşıyor ve yavaş iyileşiyorsa genelde daha çok veri sınırlı kazanç. Train çok düşük, CV yüksek ve araları açık → overfit eğilimi. Buradaki skorlar RMSE (negatif skorun eksiği alınarak).
            """
        )
    )

    cells.append(
        code(
            """
            split = int(len(best_X) * 0.8)
            X_tr, X_te = best_X.iloc[:split], best_X.iloc[split:]
            y_tr, y_te = y_all[:split], y_all[split:]
            m = res_ridge["model"]
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            plt.figure()
            plt.scatter(pred, y_te - pred, alpha=0.4)
            plt.axhline(0, color="k", lw=1)
            plt.xlabel("tahmin")
            plt.ylabel("residual")
            plt.title(f"Residual plot — örnek hedef: {viz_target}")
            plt.show()
            """
        )
    )

    cells.append(
        md(
            """
            ### Okuma rehberi — §4 Residual plot
            - **Ne işe yarar:** Tahmin hatalarının (`y - ŷ`) tahmin büyüklüğüne göre desen gösterip heteroskedasticite / sistematik yanlılık var mı bakmak.
            - **Nasıl okunur:** İdeal: noktalar `y=0` çevresinde rastgele bulut, belirgin eğri veya “fan” açılması yok. Sistematik eğri (ör. yüksek tahminde hep negatif hata) → model yapısı veya eksik özellik işareti.

            ### Neden: Linear
            - Baseline → özellik seçimi → düzenlileştirilmiş modeller sırası ödevde istendi.
            - **Not:** `ridge_path`/`lasso_path` için veri `StandardScaler` dışında gösterildi; katsayı ölçeği yorum için `Pipeline` içi katsayılarla karşılaştırılmalıdır.
            """
        )
    )

    cells.append(md("## 5. XGBoost vs LightGBM"))
    cells.append(
        code(
            """
            rows_gbm = []
            for tgt in target_columns():
                if tgt not in per_target:
                    continue
                Xb, yb = per_target[tgt]["X"], per_target[tgt]["y"]
                rx_t = randomized_xgb(Xb, yb, cv, n_iter=12)
                rl_t = randomized_lgbm(Xb, yb, cv, n_iter=12)
                per_target[tgt]["rx"] = rx_t
                per_target[tgt]["rl"] = rl_t
                xrm = (
                    float(rx_t["cv_rmse_mean"])
                    if rx_t and np.isfinite(rx_t.get("cv_rmse_mean", np.nan))
                    else np.nan
                )
                lrm = (
                    float(rl_t["cv_rmse_mean"])
                    if rl_t and np.isfinite(rl_t.get("cv_rmse_mean", np.nan))
                    else np.nan
                )
                if np.isfinite(xrm) and np.isfinite(lrm):
                    winner = "XGB" if xrm <= lrm else "LGBM"
                    gbest = min(xrm, lrm)
                elif np.isfinite(xrm):
                    winner, gbest = "XGB", xrm
                elif np.isfinite(lrm):
                    winner, gbest = "LGBM", lrm
                else:
                    winner, gbest = "", np.nan
                per_target[tgt]["tree_winner"] = winner
                per_target[tgt]["cv_tree_best"] = gbest
                rows_gbm.append(
                    {
                        "hedef": tgt,
                        "cv_xgb": xrm,
                        "cv_lgbm": lrm,
                        "tree_winner": winner,
                        "cv_tree_best": gbest,
                    }
                )
            tab_gbm = pd.DataFrame(rows_gbm)
            display(tab_gbm)

            rx = per_target[viz_target]["rx"]
            rl = per_target[viz_target]["rl"]
            display(
                pd.DataFrame(
                    [
                        {"model": "XGB", **{k: rx.get(k) for k in ["cv_rmse_mean", "cv_rmse_std"] if rx}},
                        {"model": "LGBM", **{k: rl.get(k) for k in ["cv_rmse_mean", "cv_rmse_std"] if rl}},
                    ]
                )
            )
            """
        )
    )

    cells.append(
        md(
            """
            ### Okuma rehberi — §5 XGB vs LGBM
            **`tab_gbm`**
            - **Ne işe yarar:** Her hedef için o hedefin **seçilmiş özellik matrisi** (`best_fs` sonrası `X`) üzerinde RandomizedSearch CV RMSE; hangi horizon’da hangi ağaç daha iyi.
            - **Nasıl okunur:** `tree_winner` ve `cv_tree_best` satır bazında kazanan; düşük `cv_tree_best` iyi. Alttaki iki satırlık tablo yalnızca `viz_target` (en iyi linear hataya sahip hedef) için tekrar özet.

            **İki satırlık XGB / LGBM tablosu (`viz_target`)**
            - **Ne işe yarar:** Permutation / SHAP hücreleriyle aynı veride kıyas.
            - **Nasıl okunur:** `cv_rmse_mean` düşük olan tercih; paket yoksa satır eksik kalır.
            """
        )
    )

    cells.append(
        code(
            """
            from sklearn.inspection import permutation_importance

            if rx:
                rx["model"].fit(best_X, y_all)
                pi = permutation_importance(
                    rx["model"],
                    best_X,
                    y_all,
                    scoring="neg_root_mean_squared_error",
                    n_repeats=10,
                    random_state=0,
                    n_jobs=1,
                )
                imp = (
                    pd.Series(pi.importances_mean, index=best_X.columns)
                    .sort_values(ascending=False)
                    .head(15)
                )
                display(imp.to_frame("perm_importance"))
                imp.sort_values().plot.barh(
                    figsize=(8, 6),
                    title=f"Permutation importance (XGB) — {viz_target}",
                )
                plt.show()
            """
        )
    )

    cells.append(
        md(
            """
            ### Okuma rehberi — §5 Permutation importance (tablo + yatay bar)
            - **Ne işe yarar:** Özellikleri karıştırarak model skoruna verilen marjinal katkıyı ölçer; “bu sütun gerçekten işe yarıyor mu?” sorusu.
            - **Nasıl okunur:** Üst sıradaki (yüksek `perm_importance`) özellikler skora daha duyarlı. **Dikkat:** Korelasyonlu özelliklerde önem paylaşılır; sıfıra yakın değer “gereksiz” değil “bu shuffle ile ayırt edilemedi” olabilir. Tablo ilk 15 özellik; bar grafiği aynı sıralamanın görseli.
            """
        )
    )

    cells.append(
        code(
            """
            try:
                import shap

                if rx:
                    est = rx["model"].named_steps["m"]
                    Xs = best_X.sample(min(400, len(best_X)), random_state=0)
                    explainer = shap.TreeExplainer(est)
                    sv = explainer.shap_values(Xs)
                    shap.summary_plot(sv, Xs, show=True, max_display=12)
            except Exception as e:
                print("SHAP atlandı:", e)
            """
        )
    )

    cells.append(
        md(
            """
            ### Okuma rehberi — §5 SHAP summary plot
            - **Ne işe yarar:** Tahminin özellik değerlerine göre yerel açılımını (ağaç modeli için) gösterir; yön (pozitif/negatif etki) ve büyüklük bir arada.
            - **Nasıl okunur:** Yatay eksen SHAP değeri: sağa = o gözlemde hedefi artırmaya yönelik etki. Renk: özellik değeri yüksek mü düşük mü (legend). Çoklu özellik üst üste binmiş çizgiler = o özellik sık kullanılıyor. Örnek alt küme (`sample`) kullanıldığı için tüm evrene değil, görselleştirilen alt kümeye yorum yapın.
            """
        )
    )

    cells.append(md("## 6. Single vs Multi vs Sector-featured (her hedef)"))
    cells.append(
        code(
            """
            rows_m_all = []
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import Ridge, RidgeCV
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            def ridge_cv_rmse(Xm, ym, splits=5):
                cvm = time_series_cv(splits)
                if len(Xm) < splits + 5:
                    return np.nan, np.nan
                pipe = Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                        ("m", RidgeCV(alphas=np.logspace(-3, 2, 12))),
                    ]
                )
                out = cv_eval(pipe, Xm, ym, cvm)
                return out["cv_rmse_mean"], out["cv_rmse_std"]

            for tgt in target_columns():
                if tgt not in per_target:
                    continue
                ev = events.dropna(subset=[tgt], how="any").sort_values("entry_date")
                if len(ev) < 30:
                    continue
                Xb, yb, _ = prepare_xy(ev, tgt, feats)
                Xbp = drop_high_corr_columns(Xb, 0.95)
                m0, s0 = ridge_cv_rmse(Xbp, yb)
                rows_m_all.append(
                    {"hedef": tgt, "mode": "pooled_by_entry_date", "cv_rmse_mean": m0, "cv_rmse_std": s0}
                )
                Xm2 = add_ticker_dummies(ev, Xbp)
                m2, s2 = ridge_cv_rmse(Xm2, yb)
                rows_m_all.append(
                    {"hedef": tgt, "mode": "multi_ticker_dummies", "cv_rmse_mean": m2, "cv_rmse_std": s2}
                )
                Xm3 = add_sector_peer_returns(ev, Xm2)
                m3, s3 = ridge_cv_rmse(Xm3, yb)
                rows_m_all.append(
                    {
                        "hedef": tgt,
                        "mode": "multi_dummies_sector_ret",
                        "cv_rmse_mean": m3,
                        "cv_rmse_std": s3,
                    }
                )
                singles = []
                for tk, sub in ev.groupby("ticker"):
                    Xs, ys, _ = prepare_xy(sub, tgt, feats)
                    Xs = drop_high_corr_columns(Xs, 0.95)
                    if len(sub) < 40:
                        continue
                    cvm = time_series_cv(min(5, max(2, len(sub) // 15)))
                    pipe = numeric_pipeline(Ridge(alpha=1.0))
                    sc = cross_val_score(
                        pipe,
                        Xs,
                        ys,
                        cv=cvm,
                        scoring="neg_root_mean_squared_error",
                        n_jobs=1,
                    )
                    singles.append(-float(np.mean(sc)))
                if singles:
                    mm, ss = float(np.mean(singles)), float(np.std(singles))
                    rows_m_all.append(
                        {
                            "hedef": tgt,
                            "mode": "single_per_ticker_mean_cv_rmse",
                            "cv_rmse_mean": mm,
                            "cv_rmse_std": ss,
                        }
                    )

            tab_multi = pd.DataFrame(rows_m_all)
            display(tab_multi)
            if not tab_multi.empty:
                pv = tab_multi.pivot_table(
                    index="hedef", columns="mode", values="cv_rmse_mean", aggfunc="first"
                )
                display(pv.assign(best_mode=pv.idxmin(axis=1)))
            """
        )
    )

    cells.append(
        md(
            """
            ### Okuma rehberi — §6 Multi-mode (her hedef)
            - **Ne işe yarar:** Her `y_*` için pooled / ticker dummy / sektör getirisi / ticker başına tek model stratejilerinin CV RMSE karşılaştırması.
            - **Nasıl okunur:** `tab_multi` uzun formatta; pivot tabloda `best_mode` o hedef için en düşük ortalama hatayı veren satır (mode). Bunu `summary_linear` ile yan yana düşün: biri özellik seçimi+reg, diğeri havuzlama mimarisi.
            """
        )
    )

    cells.append(md("## 7. Özet: hedef bazında seçilen modeller"))
    cells.append(
        code(
            """
            summary_all = summary_linear.merge(
                tab_gbm[["hedef", "tree_winner", "cv_tree_best"]], on="hedef", how="left"
            )
            if "tab_multi" in dir() and not tab_multi.empty:
                pv = tab_multi.pivot_table(
                    index="hedef", columns="mode", values="cv_rmse_mean", aggfunc="first"
                )
                bm = pv.idxmin(axis=1).rename("best_multi_mode").reset_index()
                bm.columns = ["hedef", "best_multi_mode"]
                summary_all = summary_all.merge(bm, on="hedef", how="left")
            display(summary_all)
            """
        )
    )

    cells.append(
        md(
            """
            ### Okuma rehberi — §7 Özet tablo
            - **Ne işe yarar:** Her hedef için §4’te seçilen FS+linear reg, §5’te ağaç kazananı ve (varsa) §6’da en iyi havuzlama modu tek satırda birleşir.
            - **Nasıl okunur:** Rapor yazarken “`y_1w` için en iyi linear `rfe`+`Lasso`, ağaçta `LGBM` kazandı” gibi cümleler kurulur. `viz_target` en düşük `cv_rmse_linear` olan hedeftir (detay grafikler onun üzerinden).
            """
        )
    )

    cells.append(md("## 8. Teslim dosyaları (Excel + PPTX)"))
    cells.append(
        code(
            """
            xlsx_path = f"data_{STUDENT_SURNAME}.xlsx"
            corr_primary = primary_corr_from_window_bundles(corr_bundles)
            corr_sheets = {
                k: v["corr"]
                for k, v in corr_bundles.items()
                if isinstance(v.get("corr"), pd.DataFrame) and not v["corr"].empty
            }
            export_excel(
                xlsx_path,
                adj,
                events,
                corr=corr_primary,
                corr_by_window=corr_sheets,
            )
            print("Yazıldı:", xlsx_path)

            bullets = [
                ("Veri ve hedef", ["yfinance panel", "7 işgünü horizon", "Özellikler giriş öncesi"]),
                ("Korelasyon", ["1–24 ay takvim pencereleri", "Isı haritası + 1y/2y clustermap", "NVDA–INTC rolling"]),
                ("Modeller", ["Linear: FS + Ridge/Lasso/ENet", "XGB vs LGBM RandomizedSearch"]),
                ("Sonuç", ["Hedef bazında FS+reg seçimi", "viz_target grafikleri", "GBM + multi özet"]),
            ]
            pptx_path = f"presentation_{STUDENT_SURNAME}.pptx"
            export_presentation(
                pptx_path,
                title="Earnings sonrası getiri tahmini",
                bullets=bullets,
            )
            print("Yazıldı:", pptx_path)
            """
        )
    )

    cells.append(
        md(
            """
            ### Okuma rehberi — §8 Excel ve PPTX
            **`data_*.xlsx`**
            - **Ne işe yarar:** Ham geniş fiyat (`adj_close_wide`), olay paneli (`events_features`), özet korelasyon (`corr_matrix`, genelde en uzun pencere) ve pencere başı korelasyon sayfaları (`corr_1m`, …).
            - **Nasıl okunur:** Özellik mühendisliği ve model için ana tablo `events_features`. Korelasyon sayfaları EDA ile tutarlı mı diye kontrol edin; sütun adları ticker sembolleri.

            **`presentation_*.pptx`**
            - **Ne işe yarar:** Sunum iskeleti; jüriye akışı anlatmak için madde madde özet.
            - **Nasıl okunur:** Slidelar otomatik metin; rakamları not defterindeki tablolarla güncellemeniz gerekir (CV RMSE, hangi model seçildi vb.).
            """
        )
    )

    cells.append(
        md(
            """
            ## Checklist (ödev tablosu)

            - [x] Korelasyon matrisi + heatmap
            - [x] Dendrogram / clustermap
            - [x] Rolling korelasyon
            - [x] Feature importance (permutation + SHAP denemesi)
            - [x] Hyperparameter tuning tabloları (RandomizedSearch sonuç nesnesi `rx`/`rl`)
            - [x] Regularization karşılaştırması
            - [x] Horizon / hedef bazında model seçimi ve özet tablo
            - [x] Stock comparison (single/multi/sector)
            """
        )
    )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = Path("code_soyad.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
