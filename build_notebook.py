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


def section_story(bridge: str, purpose: str, why_now: str, when_use: str) -> str:
    """Markdown block: Bu adımda ne yapıyoruz? (hikaye akışı)."""
    return f"""
            ### Bu adımda ne yapıyoruz?

            {bridge.strip()}

            - **Ne yapıyoruz:** {purpose.strip()}
            - **Neden şimdi:** {why_now.strip()}
            - **Ne zaman kullanılır:** {when_use.strip()}
            """


def section_reading(body: str) -> str:
    """Markdown block: Bu çıktıyı nasıl okuruz? (+ optional #### Kavram alt bölümleri)."""
    return f"""
            ### Bu çıktıyı nasıl okuruz?

            {body.strip()}
            """


def main():
    cells = []

    cells.append(
        md(
            """
            # Earnings sonrası hisse getiri tahmini

            **Dosya:** `code_soyad.ipynb` — `STUDENT_SURNAME = "soyad"` değerini kendi soyadınızla değiştirin.

            ## Hikaye

            Soru şu: **Bilanço (earnings) açıklandıktan sonra hisse ne yapar?** Her satır bir *olay*: `event_date` (yfinance kazanç tarihi), **giriş günü** `entry_date` (ertesi ilk iş günü), **özellik günü** `feature_date` (girişten bir iş günü önce — sızıntıyı önlemek için). **Hedefler** `y_*`: giriş kapanışından 1, 3, 5, … 63 iş günü sonrasına kadar basit getiri.

            Önce veriyi kuruyoruz, ortak hareketi ve hedef dağılımlarını anlıyoruz, sonra **zaman serisi çapraz doğrulama (cross-validation, CV)** ile modelleri dürüstçe kıyaslıyoruz. Kavramlar tek listede değil; **ilk kullanıldıkları adımda** tanımlanır (Türkçe anlatım, teknik terimler parantezde İngilizce).

            ## Yol haritası

            1. **§1 Veri** — olay paneli, tarih sırası  
            2. **§2 Korelasyon** — hisseler birlikte mi hareket ediyor?  
            3. **§3 Özellikler / hedefler** — neyi tahmin ediyoruz, dağılım nasıl?  
            4. **§3.5 Zaman serisi CV** — geçmişe eğit, gelecek fold’da ölç (tüm modeller için ortak kural)  
            5. **§4 Regresyon** — özellik seçimi + Ridge / Lasso / ElasticNet  
            6. **§4.5 Sınıflandırma** — LDA / QDA / Naive Bayes: beat ve getiri yönü  
            7. **§5–§6** — XGB / LGBM ve havuzlama modları  
            8. **§7–§9** — özet, tahmin tabloları, Excel / sunum  

            **Pratik:** Fiyatlar `data/raw_prices_long.csv`, kazançlar `data/earnings_dates_cache.csv` önbelleğe alınır. Evren [tickers.csv](tickers.csv); yoksa varsayılan oluşturulur. Piyasa/sektör: ETF yok, eşit ağırlıklı sepet. EPS eksikleri NaN kalabilir.
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
                configure_plot_style,
                correlation_bundles_by_calendar_windows,
                drop_high_corr_columns,
                feature_columns,
                primary_corr_from_window_bundles,
                slice_returns_calendar_tail,
                target_columns,
                classification_feature_columns,
                make_direction_label,
            )
            from IPython.display import Markdown, display

            from odev_modeling import (
                add_sector_peer_returns,
                add_ticker_dummies,
                classification_pipeline,
                correlation_prune_cv,
                cv_eval,
                cv_eval_classification,
                cv_fold_sizes,
                discriminant_bundle,
                format_sonuc,
                grid_regularized,
                linear_baseline_cv,
                materialize_fs_matrix,
                numeric_pipeline,
                pick_viz_target,
                plot_all_cv_folds,
                prepare_xy_class,
                prepare_xy_with_meta,
                randomized_lgbm,
                randomized_xgb,
                rfe_cv,
                select_k_best_cv,
                teach_cv_compare,
                teach_cv_compare_classify,
                gbm_permutation_importance,
                time_series_cv,
                prepare_xy,
            )
            from sklearn.linear_model import Ridge, Lasso
            from sklearn.model_selection import cross_val_score, learning_curve

            from odev_export import export_excel, export_presentation

            sns.set_theme(style="whitegrid", context="talk")
            configure_plot_style(dpi=200, savefig_dpi=400, retina=True)
            """
        )
    )

    cells.append(md("## 1. Veri çekme ve panel"))
    cells.append(
        md(
            section_story(
                bridge="Başlangıç: tahmin yapmadan önce **olay tablosunun** (`events`) güvenilir olduğundan emin olmalıyız.",
                purpose="yfinance’tan hizalanmış fiyat paneli ve ticker başına kazanç tarihleriyle **olay × özellik × hedef** satırları üretmek; boyut ve örnek satırları görmek.",
                why_now="Model skoru anlamsız olur, eğer tarihler karışıksa veya örnek sayısı çok düşükse — CV’yi §3.5’te kuracağız.",
                when_use="Her yeni veri kaynağı, ticker listesi veya önbellek yenilemesinden sonra bu adımı tekrarlayın.",
            )
        )
    )
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
        code(
            """
            ev_chrono = events.sort_values("entry_date")
            tab_veri = pd.DataFrame(
                [
                    {
                        "n_olay": len(events),
                        "n_ticker": events["ticker"].nunique(),
                        "tarih_min": ev_chrono["entry_date"].min(),
                        "tarih_max": ev_chrono["entry_date"].max(),
                    }
                ]
            )
            display(tab_veri)
            ev_chrono.groupby(ev_chrono["entry_date"].dt.to_period("M")).size().plot(
                kind="bar", figsize=(12, 4), title="Aylık olay sayısı (entry_date)"
            )
            plt.tight_layout()
            plt.show()
            _demo_tgt = [c for c in target_columns() if c in events.columns][0]
            y_raw = events[_demo_tgt].values
            y_sorted = ev_chrono[_demo_tgt].values
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
            axes[0].plot(y_raw, ".", alpha=0.5, markersize=3)
            axes[0].set_title(f"Sıra: ticker→tarih ({_demo_tgt})")
            axes[1].plot(y_sorted, ".", alpha=0.5, markersize=3)
            axes[1].set_title(f"Sıra: entry_date ({_demo_tgt})")
            plt.tight_layout()
            plt.show()
            sonuc_1 = format_sonuc(
                "1 Veri",
                [
                    f"Panel hazır: {len(events)} olay, {events['ticker'].nunique()} ticker; tarih aralığı {tab_veri['tarih_min'].iloc[0]} – {tab_veri['tarih_max'].iloc[0]}.",
                    "Hikaye devamı §2: hisseler ortak mı hareket ediyor — korelasyon (correlation) bakacağız.",
                    "Unutmayın: CV için satırlar **entry_date** sırasında olmalı (sağ grafik); ticker’a göre sıra yanıltıcıdır.",
                ],
            )
            display(Markdown(sonuc_1))
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            - **`events.shape` / `adj.shape`:** Satır = kazanç olayı sayısı; `adj` sütunları = indirilen hisse. Çok az satır → sonraki CV güvenilmez.
            - **`events.head()`:** `feature_date` girişten önce olmalı; `y_*` uç değerler aykırı olay olabilir.
            - **Aylık olay grafiği:** Zaman içinde veri yoğunluğu; seyrek aylar fold’ları küçültür.
            - **Ticker sırası vs `entry_date` sırası:** Sağ panel geleceği “görmez”; sol panel CV’yi bozar — §3.5’te hep kronolojik sıra kullanacağız.

            Önbellek: `force_refresh_raw=True` veya `force_refresh_earnings=True` ile yenileme; ikinci çalıştırma genelde hızlıdır.
                """
            )
        )
    )

    cells.append(md("## 2. Korelasyon analizi"))
    cells.append(
        md(
            section_story(
                bridge="§1’de paneli kurduk; şimdi hisseler **birlikte mi hareket ediyor** ona bakıyoruz.",
                purpose="Farklı takvim pencerelerinde (1 ay – 2 yıl) getiri **korelasyon (correlation)** matrisleri, ısı haritası, küme haritası (clustermap).",
                why_now="Yüksek korelasyon → özellikler birbirinin kopyası gibi; §4’te özellik seçimi (feature selection) ve sektör sinyalleri bunun için anlamlı.",
                when_use="Evren genişlediğinde veya piyasa rejimi değiştiğinde korelasyon yapısını yeniden kontrol edin.",
            )
        )
    )
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
                fig, ax = plt.subplots(figsize=(14, 11), dpi=200)
                sns.heatmap(cmat, ax=ax, cmap="vlag", center=0, square=True)
                ax.set_title(f"Getiri korelasyon — {ttl}")
                plt.tight_layout()
                plt.show()

            for k in ("1y", "2y"):
                meta = corr_bundles.get(k)
                if not meta or meta["corr"].empty or meta["corr"].shape[0] < 2:
                    continue
                g = sns.clustermap(
                    meta["corr"], cmap="vlag", center=0, figsize=(14, 14)
                )
                g.fig.suptitle(
                    f"Clustermap — {CORR_WINDOW_TITLE_TR.get(k, k)}", y=1.02
                )
                plt.show()

            if "NVDA" in ret_w.columns and "INTC" in ret_w.columns:
                plt.figure(figsize=(13, 6), dpi=200)
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

            corr_density = []
            for k in ("1m", "1y"):
                meta = corr_bundles.get(k)
                if not meta or meta["corr"].empty:
                    continue
                c = meta["corr"].values.copy()
                np.fill_diagonal(c, np.nan)
                corr_density.append(
                    {
                        "pencere": CORR_WINDOW_TITLE_TR.get(k, k),
                        "mean_abs_corr": float(np.nanmean(np.abs(c))),
                    }
                )
            tab_corr_cmp = pd.DataFrame(corr_density)
            display(tab_corr_cmp)
            if len(tab_corr_cmp) >= 2:
                tab_corr_cmp.set_index("pencere")["mean_abs_corr"].plot.bar(
                    figsize=(8, 4), title="Ortalama |ρ| (köşegen hariç)"
                )
                plt.tight_layout()
                plt.show()
            _win_best = (
                tab_corr_cmp.loc[tab_corr_cmp["mean_abs_corr"].idxmax(), "pencere"]
                if len(tab_corr_cmp)
                else "—"
            )
            display(
                Markdown(
                    format_sonuc(
                        "2 Korelasyon",
                        [
                            f"Ortak hareket en yoğun pencere: **{_win_best}** — sektör/piyasa özellikleri §4–§6’da işe yarayabilir.",
                            "§3’e geçiyoruz: hedef getirilerin dağılımı gürültülü mü, hangi horizon zor?",
                            "Çoklu doğrusal bağlantı (multicollinearity) riski → §4’te `corr_prune` ve diğer FS yöntemleri.",
                        ],
                    )
                )
            )
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            - **Pencere tablosu:** `n_gün` çok düşükse o pencerenin ısı haritasına temkinli yaklaşın.
            - **Isı haritası:** Kırmızı/mavi bloklar = ortak faktör; §6’da sektör sepeti özelliği bunu açıkça modele verir.
            - **Clustermap:** Benzer profilli hisseler aynı dalda — raporda “küme” diye özetlenebilir.
            - **NVDA–INTC rolling:** İlişki zamanla değişir; mutlak seviye yerine göreli sıraya bakın.

            #### Kavram: çoklu doğrusal bağlantı (multicollinearity)
            Özellikler birbirine çok benziyorsa linear model katsayıları kararsızlaşır. **Ne zaman:** Korelasyon matrisinde |ρ| yüksek çiftler çoksa. **Ne yaparız:** §4’te `corr_prune`, Ridge cezası veya daha az özellik.
                """
            )
        )
    )

    cells.append(md("## 3. Özellikler ve hedef dağılımları"))
    cells.append(
        md(
            section_story(
                bridge="Ortak hareketi gördük; şimdi **neyi tahmin ettiğimizi** ve özellik kümesinin boyutunu netleştiriyoruz.",
                purpose="Tüm `y_*` horizon’ları için histogram; özellik sayısı; kısa/uzun horizon yayılımı (std, çarpıklık).",
                why_now="Gürültülü veya çarpık hedefler RMSE’yi şişirir; hangi model ailesinin deneneceğine ipucu verir.",
                when_use="Yeni hedef tanımı veya horizon eklendiğinde bu adımı tekrarlayın.",
            )
        )
    )
    cells.append(
        code(
            """
            feats = feature_columns(events)
            print("Özellik sayısı:", len(feats))

            for tcol in target_columns():
                plt.figure(figsize=(7, 4.5), dpi=200)
                events[tcol].dropna().hist(bins=40)
                plt.title(f"Hedef {tcol}")
                plt.tight_layout()
                plt.show()

            tgt_stats = []
            for tcol in target_columns():
                s = events[tcol].dropna()
                if s.empty:
                    continue
                tgt_stats.append(
                    {
                        "hedef": tcol,
                        "std": float(s.std()),
                        "skew": float(s.skew()),
                    }
                )
            tab_tgt = pd.DataFrame(tgt_stats)
            display(tab_tgt)
            if not tab_tgt.empty:
                short_h = [c for c in tab_tgt["hedef"] if c in ("y_1d", "y_3d")]
                long_h = [c for c in tab_tgt["hedef"] if c in ("y_1m", "y_3m")]
                std_short = tab_tgt.loc[tab_tgt["hedef"].isin(short_h), "std"].mean()
                std_long = tab_tgt.loc[tab_tgt["hedef"].isin(long_h), "std"].mean()
                display(
                    Markdown(
                        format_sonuc(
                            "3 Özellikler",
                            [
                                f"Kısa horizon std ≈ {std_short:.4f}, uzun ≈ {std_long:.4f} — uzun vade genelde daha geniş (gürültülü) hedef.",
                                f"{len(feats)} özellik var; hedef başına en iyi kombinasyonu henüz bilmiyoruz.",
                                "Sırada §3.5: skoru hesaplamadan önce zamanı doğru bölmeyi (CV) öğreneceğiz.",
                            ],
                        )
                    )
                )
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            - **Histogramlar:** Tek tepeli/simetrik → linear regresyon makul; uzun kuyruk → aykırılar RMSE’yi etkiler (§4’te tanımlanacak).
            - **`tab_tgt`:** `std` ve `skew` horizon’lar arası kıyas için; çok çarpık hedefte ağaç modelleri (§5) denemeye değer.
            - **Özellik sayısı:** Olay sayısının çok üstündeyse overfit riski — §4’te özellik seçimi (feature selection) şart.
                """
            )
        )
    )

    cells.append(md("## 3.5 Zaman serisi CV — öğrenme yolu"))
    cells.append(
        md(
            section_story(
                bridge="Hedefleri tanıdık; şimdi **nasıl ölçeceğimizi** öğreniyoruz — rastgele train/test burada yanıltıcıdır.",
                purpose="Blok zaman serisi CV (block time-series CV): geçmiş fold’larda eğit, sonraki blokta skorla; `gap`, `offset`, gecikmeli özellik demosu; global `cv` nesnesini seçmek.",
                why_now="§4–§8’deki tüm modeller aynı `cv` ile kıyaslanacak; adil karşılaştırma ancak böyle.",
                when_use="Zaman sıralı panel veride (finans, sensör, satış) her zaman zaman serisi CV; i.i.d. varsayımı olan `KFold` değil.",
            )
            + """

            #### Kavram: zaman serisi çapraz doğrulama (time-series cross-validation, CV)

            - **Ne:** Veriyi `entry_date` sırasında dilimleyip modeli birkaç kez “geçmişe eğit, ileri dilimi test et” çalıştırmak ([makale özeti](https://emad-ezzeldin4.medium.com/introduction-to-time-series-cross-validation-mastering-predictive-accuracy-in-sequential-data-139710a07915)).
            - **`gap`:** Eğitim ile test arasında boşluk (embargo).
            - **`offset`:** İlk test fold’undan önce minimum geçmiş uzunluğu.
            - **Nasıl okuruz:** `cv_rmse_mean` (regresyon) veya `cv_score_mean` (sınıflandırma) — düşük hata / yüksek AUC iyi; `*_std` yüksekse fold’lar arası tutarsızlık.
            """
        )
    )
    cells.append(
        code(
            """
            _cv_demo_tgt = [c for c in target_columns() if c in events.columns][0]
            X_cv, y_cv, _ = prepare_xy(events, _cv_demo_tgt, feats)
            x_cv = np.arange(len(y_cv))
            pipe_cv = numeric_pipeline(Ridge(alpha=1.0))

            cv_base = time_series_cv(5, gap=0, offset=0)
            display(cv_fold_sizes(cv_base, len(y_cv)))
            plot_all_cv_folds(X_cv, y_cv, pipe_cv, cv_base, x_series=x_cv)

            cv_gap = time_series_cv(5, gap=1, offset=0)
            tab_gap0 = teach_cv_compare([("gap=0", pipe_cv)], X_cv, y_cv, cv_base)
            tab_gap1 = teach_cv_compare([("gap=1", pipe_cv)], X_cv, y_cv, cv_gap)
            tab_gap = pd.concat([tab_gap0, tab_gap1], ignore_index=True)
            display(tab_gap)

            lag_cols = [c for c in feats if "lag" in c or "prior_post" in c or "surprise_lag" in c]
            feats_nolag = [c for c in feats if c not in lag_cols]
            X_nolag, y_nl, _ = prepare_xy(events, _cv_demo_tgt, feats_nolag)
            X_lag, y_lg, _ = prepare_xy(events, _cv_demo_tgt, feats)
            tab_lag = teach_cv_compare(
                [
                    ("lagsiz", pipe_cv),
                ],
                X_nolag,
                y_nl,
                cv_base,
            )
            tab_lag2 = teach_cv_compare([("gecikmeli", pipe_cv)], X_lag, y_lg, cv_base)
            display(pd.concat([tab_lag, tab_lag2], ignore_index=True))

            off = max(10, len(y_cv) // 10)
            cv_off = time_series_cv(5, gap=0, offset=off)
            display(cv_fold_sizes(cv_off, len(y_cv)))
            plot_all_cv_folds(X_cv, y_cv, pipe_cv, cv_off, x_series=x_cv)

            cv = time_series_cv(5, gap=0, offset=0)
            _g0 = tab_gap0["cv_rmse_mean"].iloc[0] if len(tab_gap0) else np.nan
            _g1 = tab_gap1["cv_rmse_mean"].iloc[0] if len(tab_gap1) else np.nan
            _use_gap = 1 if np.isfinite(_g0) and np.isfinite(_g1) and _g1 <= _g0 else 0
            if _use_gap:
                cv = time_series_cv(5, gap=1, offset=0)
            display(
                Markdown(
                    format_sonuc(
                        "3.5 Zaman serisi CV",
                        [
                            f"Artık ortak kural net: `cv` = n_splits=5, gap={cv.gap}, offset={cv.offset} (demo: `{_cv_demo_tgt}`).",
                            "Fold grafiği: sol eksen gerçek getiri (mavi=eğitim, yeşil=test); sağ eksen kırmızı=test tahmini (ayrı ölçek).",
                            "§4’te regresyon ve RMSE ile devam; aynı CV nesnesi değişmeden kalacak.",
                        ],
                    )
                )
            )
            """
        )
    )

    cells.append(md("## 4. Linear pipeline — her hedef için en iyi FS + reg"))
    cells.append(
        md(
            section_story(
                bridge="CV kuralını §3.5’te kabul ettik; şimdi **sürekli getiri** tahmini için linear modelleri deniyoruz.",
                purpose="Her `y_*` için özellik seçimi (FS) + Ridge / Lasso / ElasticNet; `tab_fs_all`, `summary_linear`, tanı grafikleri.",
                why_now="Regresyon doğrudan getiri büyüklüğünü verir; beat veya yön sorularından farklı (§4.5).",
                when_use="Hedef sürekli ve kabaca doğrusal ilişkiler arıyorsanız; güçlü doğrusal olmayanlık için §5 ağaçlara bakın.",
            )
            + """

            #### Kavram: RMSE (root mean squared error)

            Tahmin hatalarının kare ortalamasının karekökü; **hedefle aynı birim** (getiri). **Düşük RMSE = daha iyi.** Aykırı hatalara duyarlıdır. Tablolarda `cv_rmse_mean` / `cv_rmse_std`: fold ortalaması ve dalgalanma.
            """
        )
    )
    cells.append(
        code(
            """
            if "cv" not in dir():
                cv = time_series_cv(5, gap=0, offset=0)
            per_target = {}


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
                best_X = materialize_fs_matrix(best_fs, X_all, y_all, bundles)
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

            REG_KEY = {"Ridge": "res_ridge", "Lasso": "res_lasso", "ElasticNet": "res_enet"}
            viz_target = pick_viz_target(per_target, fallback=next(iter(per_target), None))
            print("Görselleştirme (viz_target):", viz_target)
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

            ridge_lasso_kiyaslama = pd.DataFrame(
                [
                    {
                        "hedef": t,
                        "cv_ridge": per_target[t]["res_ridge"]["cv_rmse_mean"],
                        "cv_lasso": per_target[t]["res_lasso"]["cv_rmse_mean"],
                        "std_ridge": per_target[t]["res_ridge"]["cv_rmse_std"],
                        "std_lasso": per_target[t]["res_lasso"]["cv_rmse_std"],
                    }
                    for t in per_target
                ]
            )
            ridge_lasso_kiyaslama["fark_lasso_minus_ridge"] = (
                ridge_lasso_kiyaslama["cv_lasso"] - ridge_lasso_kiyaslama["cv_ridge"]
            )
            ridge_lasso_kiyaslama["kazanan"] = np.where(
                ridge_lasso_kiyaslama["cv_ridge"] <= ridge_lasso_kiyaslama["cv_lasso"],
                "Ridge",
                "Lasso",
            )
            display(ridge_lasso_kiyaslama.sort_values("hedef"))
            print(
                "Özet: Ridge kazanan hedef sayısı:",
                int((ridge_lasso_kiyaslama["kazanan"] == "Ridge").sum()),
                "| Lasso:",
                int((ridge_lasso_kiyaslama["kazanan"] == "Lasso").sum()),
            )

            plt.figure(figsize=(12, 6), dpi=200)
            x = np.arange(len(ridge_lasso_kiyaslama))
            w = 0.35
            rl_df = ridge_lasso_kiyaslama.sort_values("hedef").reset_index(drop=True)
            plt.bar(x - w / 2, rl_df["cv_ridge"], width=w, label="Ridge")
            plt.bar(x + w / 2, rl_df["cv_lasso"], width=w, label="Lasso")
            plt.xticks(x, rl_df["hedef"], rotation=45, ha="right")
            plt.ylabel("CV RMSE")
            plt.title("Ridge vs Lasso — tüm hedefler (her hedefin seçilmiş FS matrisi)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            tab_viz_fs = per_target[viz_target]["tab_fs"]
            tab_viz_fs.set_index("yöntem")["cv_rmse_mean"].plot.bar(
                figsize=(10, 4), title=f"FS karşılaştırma — {viz_target}"
            )
            plt.ylabel("CV RMSE")
            plt.tight_layout()
            plt.show()
            top2 = tab_viz_fs.nsmallest(2, "cv_rmse_mean")["yöntem"].tolist()
            for fs_name in top2[:2]:
                Xm = materialize_fs_matrix(
                    fs_name, per_target[viz_target]["X"], per_target[viz_target]["y"], per_target[viz_target]["bundles"]
                )
                plot_all_cv_folds(
                    Xm,
                    per_target[viz_target]["y"],
                    per_target[viz_target]["bundles"][fs_name]["model"],
                    cv,
                    x_series=np.arange(len(per_target[viz_target]["y"])),
                )
            display(
                Markdown(
                    format_sonuc(
                        "4a Özellik seçimi",
                        [
                            f"Her horizon için bir FS kazandı — örnek hedef `{viz_target}`, en iyi iki yöntem: {', '.join(top2)}.",
                            "Hikaye: önce özellik kümesi, sonra regresyon tipi; hedefler birbirinden farklı FS isteyebilir.",
                            "Detay fold grafiği yalnızca `viz_target` için (performans).",
                        ],
                    )
                )
            )
            _win_reg = per_target[viz_target]["best_reg"]
            plot_all_cv_folds(
                best_X,
                y_all,
                per_target[viz_target][REG_KEY[_win_reg]]["model"],
                cv,
                x_series=np.arange(len(y_all)),
            )
            display(
                Markdown(
                    format_sonuc(
                        "4b Düzenlileştirme",
                        [
                            f"`{viz_target}` için en iyi düzenlileştirme: **{_win_reg}** (Ridge / Lasso / ElasticNet üçlüsünden).",
                            f"Tüm horizon’larda Ridge {int((ridge_lasso_kiyaslama['kazanan'] == 'Ridge').sum())}, Lasso {int((ridge_lasso_kiyaslama['kazanan'] == 'Lasso').sum())} kez önde.",
                            "Sırada §4.5: aynı veriyle **sınıflandırma** (beat ve yön) — farklı soru, farklı metrik (ROC-AUC).",
                        ],
                    )
                )
            )
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            - **`tab_fs_all`:** Her hedefte dört FS yöntemi; en düşük `cv_rmse_mean` → `best_fs`.

            #### Kavram: özellik seçimi (feature selection)
            - **baseline:** Tüm özellikler — referans.
            - **corr_prune:** Yüksek korelasyonlu sütunları atar (§2 ile bağlantılı).
            - **kbest:** İstatistiksel olarak en ilişkili ilk *k* özellik.
            - **RFE (recursive feature elimination):** Ridge ile adım adım zayıf özellikleri eleyerek alt küme; maliyetli, taban modele bağlı.

            - **`summary_linear`:** Seçilen FS + `best_reg` + `cv_rmse_linear`; rapor cümlesi buradan kurulur.
            - **`ridge_lasso_kiyaslama`:** Aynı FS üzerinde Ridge vs Lasso; `kazanan` sütunu hızlı okuma.
            - **CV fold grafiği:** Sol = gerçek getiri; sağ (kırmızı) = test tahmini — tahmin sıfıra yakın görünse bile sağ eksende ölçeği kontrol edin.

            #### Kavram: Ridge, Lasso, ElasticNet
            - **Ridge:** L2 cezası — katsayıları küçültür, çoklu doğrusal bağlantıda stabil.
            - **Lasso:** L1 cezası — bazı katsayıları sıfırlar (seyrek model).
            - **ElasticNet:** L1+L2 karışımı. **Ne zaman:** Tabloda hangi ceza düşük CV RMSE veriyorsa o horizon için.
                """
            )
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
            section_reading(
                """
            **Katsayı yolları (Ridge / Lasso):** `alpha` arttıkça ceza güçlenir; Lasso’da çizginin eksene düşmesi = özellik elendi.
            Ham X ile çizildi — büyüklükleri doğrudan “önem sırası” değil; pipeline içi ölçekli katsayılarla kıyaslayın.
                """
            )
        )
    )

    cells.append(
        code(
            """
            _lc_gap = np.nan
            for _lc_name, _lc_model in (
                ("Ridge", res_ridge["model"]),
                ("Lasso", res_lasso["model"]),
                ("ElasticNet", res_enet["model"]),
            ):
                train_sizes, train_scores, test_scores = learning_curve(
                    _lc_model,
                    best_X,
                    y_all,
                    cv=cv,
                    scoring="neg_root_mean_squared_error",
                    train_sizes=np.linspace(0.2, 1.0, 5),
                    n_jobs=1,
                )
                train_rmse = -train_scores
                test_rmse = -test_scores
                if _lc_name == "Ridge":
                    _lc_gap = float(test_rmse.mean(axis=1)[-1] - train_rmse.mean(axis=1)[-1])
                plt.figure(figsize=(11, 5), dpi=200)
                plt.plot(train_sizes, train_rmse.mean(axis=1), label="train")
                plt.plot(train_sizes, test_rmse.mean(axis=1), label="cv")
                plt.legend()
                plt.xlabel("Eğitim örnek sayısı")
                plt.ylabel("RMSE")
                plt.title(
                    f"Learning curve ({_lc_name}) — {viz_target}"
                )
                plt.tight_layout()
                plt.show()
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            **Learning curve (öğrenme eğrisi):** Eğitim boyutu arttıkça train vs CV RMSE. Eğriler yakınsıyorsa ek veri sınırlı fayda; train düşük CV yüksek → **aşırı uyum (overfitting)** eğilimi.
                """
            )
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

            _lc_gap = float(test_rmse.mean(axis=1)[-1] - train_rmse.mean(axis=1)[-1])
            display(
                Markdown(
                        format_sonuc(
                        "4c Tanı grafikleri",
                        [
                            f"`{viz_target}`: train–CV RMSE farkı ≈ {_lc_gap:.4f} — overfit mi, veri doygun mu buradan okunur.",
                            "Residual grafiği rastgele bulut ise sistematik yanlılık zayıf.",
                            "Linear hikaye burada; doğrusal olmayan desenler için §5 XGB/LGBM.",
                        ],
                    )
                )
            )
            """
        )
    )

    cells.append(md("## 4.5 Discriminant analizi ve Naive Bayes (zaman serisi CV)"))
    cells.append(
        md(
            section_story(
                bridge="§4 getiri **büyüklüğünü** tahmin etti; şimdi **ikili sorular**: kazanç beat mi? Getiri pozitif mi?",
                purpose="LDA, QDA, Gaussian Naive Bayes ile `eps_beat` ve her horizon için yön (`y_* > 0`); ROC-AUC (CV).",
                why_now="Beat ile fiyat yönü farklı bilgiler; yatırım kararı için ikisini ayırmak gerekir.",
                when_use="Hedef sınıf (0/1) ve özellikler sürekli olduğunda; LDA/QDA için ölçekleme şart (pipeline’da var).",
            )
            + """

            #### Kavram: LDA, QDA, Naive Bayes ve ROC-AUC

            - **LDA (linear discriminant analysis):** Ortak kovaryans, linear sınır.
            - **QDA (quadratic discriminant analysis):** Sınıf başına kovaryans, daha esnek; az veride kararsız olabilir.
            - **Gaussian Naive Bayes:** Özellikler koşullu bağımsız varsayımı; hızlı baseline.
            - **ROC-AUC:** 0.5 = rastgele, 1 = mükemmel ayrım; **yüksek iyi.** `eps_beat` ve EPS sütunları özelliklerden çıkarılır (sızıntı önleme).
            """
        )
    )
    cells.append(
        code(
            """
            if "cv" not in dir():
                cv = time_series_cv(5, gap=0, offset=0)
            clf_models = discriminant_bundle()
            feats_beat = classification_feature_columns(events, "beat")
            X_beat, y_beat, _ = prepare_xy_class(events, "eps_beat", feats_beat)

            tab_balance = (
                pd.Series(y_beat, name="eps_beat")
                .value_counts()
                .rename_axis("sinif")
                .reset_index(name="adet")
            )
            tab_balance["oran"] = tab_balance["adet"] / tab_balance["adet"].sum()
            display(tab_balance)
            tab_balance.set_index("sinif")["adet"].plot.bar(
                figsize=(7, 4.5), title="eps_beat sınıf dengesi"
            )
            plt.ylabel("adet")
            plt.tight_layout()
            plt.show()

            specs_beat = [(n, clf_models[n]) for n in clf_models]
            tab_clf_beat = teach_cv_compare_classify(specs_beat, X_beat, y_beat, cv)
            display(tab_clf_beat)
            tab_clf_beat.set_index("yöntem")["cv_score_mean"].plot.bar(
                figsize=(8, 4.5), title="ROC-AUC (CV) — eps_beat"
            )
            plt.ylabel("ROC-AUC")
            plt.tight_layout()
            plt.show()

            _beat_winner = str(tab_clf_beat.loc[tab_clf_beat["cv_score_mean"].idxmax(), "yöntem"])
            _best_beat_pipe = clf_models[_beat_winner]
            teach_cv_compare_classify(
                [(_beat_winner, _best_beat_pipe)],
                X_beat,
                y_beat,
                cv,
                plot=True,
                x_series=np.arange(len(y_beat)),
            )
            clf_beat_auc = float(tab_clf_beat["cv_score_mean"].max())
            display(
                Markdown(
                        format_sonuc(
                        "4.5a eps_beat",
                        [
                            f"Beat tahmini: **{_beat_winner}** önde (CV ROC-AUC ≈ {clf_beat_auc:.3f}) — 0.5 üstü sinyal var, mükemmel değil.",
                            "Beat iyi tahmin etmek, hissenin yükselacağı anlamına gelmez; işlem kararı için §4 getiri tablolarına da bakın.",
                            "Yön analizi §4.5b’de tüm horizon’lar için tabloda.",
                        ],
                    )
                )
            )

            rows_dir = []
            feats_dir = classification_feature_columns(events, "direction")
            for h in target_columns():
                ev_h = events.copy()
                ev_h["_dir"] = make_direction_label(events, h)
                Xd, yd, _ = prepare_xy_class(ev_h, "_dir", feats_dir)
                if len(yd) < 30 or len(np.unique(yd)) < 2:
                    continue
                tab_h = teach_cv_compare_classify(specs_beat, Xd, yd, cv)
                for _, r in tab_h.iterrows():
                    rows_dir.append(
                        {
                            "hedef": h,
                            "yöntem": r["yöntem"],
                            "cv_score_mean": r["cv_score_mean"],
                            "cv_score_std": r["cv_score_std"],
                        }
                    )
            tab_clf_direction = pd.DataFrame(rows_dir)
            display(tab_clf_direction)

            if not tab_clf_direction.empty:
                pv_auc = tab_clf_direction.pivot_table(
                    index="hedef", columns="yöntem", values="cv_score_mean", aggfunc="first"
                )
                display(pv_auc.assign(best_clf=pv_auc.idxmax(axis=1)))
                viz_horizon = str(pv_auc.max(axis=1).idxmax())
                clf_dir_winner = str(pv_auc.loc[viz_horizon].idxmax())
                clf_dir_auc = float(pv_auc.loc[viz_horizon].max())
                pv_auc.max(axis=1).plot.bar(
                    figsize=(11, 4.5), title="En iyi model ROC-AUC (horizon başına)"
                )
                plt.ylabel("max ROC-AUC")
                plt.tight_layout()
                plt.show()
                ev_v = events.copy()
                ev_v["_dir"] = make_direction_label(events, viz_horizon)
                Xv, yv, _ = prepare_xy_class(ev_v, "_dir", feats_dir)
                teach_cv_compare_classify(
                    [(clf_dir_winner, clf_models[clf_dir_winner])],
                    Xv,
                    yv,
                    cv,
                    plot=True,
                    x_series=np.arange(len(yv)),
                )
                display(
                    Markdown(
                        format_sonuc(
                            "4.5b yön",
                            [
                                f"Getiri yönü en ayırt edilebilir horizon: **{viz_horizon}**; model **{clf_dir_winner}** (AUC ≈ {clf_dir_auc:.3f}).",
                                "Canlı ticaret için tek başına yeterli değil — CV metrikleri + maliyet + rejim değişimi gerekir.",
                                "§5: aynı özelliklerle doğrusal olmayan regresyon (XGB / LGBM).",
                            ],
                        )
                    )
                )
            else:
                viz_horizon = ""
                clf_dir_winner = ""
                clf_dir_auc = float("nan")
                display(
                    Markdown(
                        format_sonuc(
                            "4.5b yön",
                            ["Yeterli etiketli satır yok; yön sınıflandırması atlandı."],
                        )
                    )
                )
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            - **`tab_clf_beat` / çubuk grafik:** Üç sınıflandırıcı kıyası; tek fold’da tek sınıf varsa o fold atlanabilir.
            - **`tab_clf_direction` + pivot:** Horizon başına hangi model (LDA/QDA/NB) daha iyi.
            - **Fold grafiği (beat + bir horizon):** Sınıf etiketleri 0/1 — regresyon fold’larından farklı okuma.
                """
            )
        )
    )

    cells.append(md("## 5. XGBoost vs LightGBM"))
    cells.append(
        md(
            section_story(
                bridge="Linear ve sınıflandırma tablolarını gördük; şimdi **ağaç toplulukları (tree ensembles)** ile getiri regresyonunu güçlendirmeyi deniyoruz.",
                purpose="Her hedef için seçilmiş FS matrisi üzerinde XGBoost (XGB) ve LightGBM (LGBM) RandomizedSearch; permutation importance ve (varsa) SHAP.",
                why_now="Doğrusal olmayan etkileşim ve eşik davranışları için; CV RMSE ile §4’teki linear kıyaslanır.",
                when_use="Büyük tabular veri, doğrusal model yetersiz kaldığında; yorum için permutation / SHAP (dikkat: korelasyonlu özelliklerde paylaşımlı önem).",
            )
            + """

            #### Kavram: gradient boosting (XGB, LGBM)

            Ardışık karar ağaçları önceki hataları düzeltir. **XGB** ve **LGBM** aynı aile; hangi horizon’da CV RMSE düşükse o kazanır — evrensel üstünlük yok.
            """
        )
    )
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
            if rx and rl:
                teach_cv_compare(
                    [("XGB", rx["model"]), ("LGBM", rl["model"])],
                    best_X,
                    y_all,
                    cv,
                    plot=True,
                    x_series=np.arange(len(y_all)),
                )
            _xgb_w = int((tab_gbm["tree_winner"] == "XGB").sum())
            _lgb_w = int((tab_gbm["tree_winner"] == "LGBM").sum())
            _vt_lin = per_target[viz_target]["cv_best_lin"]
            _vt_tree = per_target[viz_target].get("cv_tree_best", np.nan)
            display(
                Markdown(
                    format_sonuc(
                        "5 XGB / LGBM",
                        [
                            f"Ağaç kazananları: XGB {_xgb_w} hedef, LGBM {_lgb_w} hedef — horizon’a göre değişir.",
                            f"`{viz_target}`: linear {_vt_lin:.4f} vs en iyi ağaç {_vt_tree:.4f} RMSE — hangisi düşükse o veri için tercih.",
                            "§6: aynı hedefler için **havuzlama** (pooled vs ticker dummy vs sektör).",
                        ],
                    )
                )
            )
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            - **`tab_gbm`:** `tree_winner`, `cv_tree_best` — düşük RMSE iyi.
            - **Permutation importance:** Özelliği karıştırınca skor ne kadar düşüyor; korelasyonlu özelliklerde paylaşımlı önem.
            - **SHAP:** Yerel etki yönü ve büyüklüğü; alt örneklem (`sample`) üzerinde — evrensel iddia değil.
                """
            )
        )
    )

    cells.append(
        code(
            """
            if rx:
                imp, pi_note = gbm_permutation_importance(
                    rx["model"], best_X, y_all, n_repeats=10, random_state=0, top_n=15
                )
                if pi_note:
                    print(pi_note)
                display(imp.to_frame("perm_importance"))
                if imp.max() > 0:
                    imp.sort_values().plot.barh(
                        figsize=(10, 7),
                        title=f"Permutation importance (XGB) — {viz_target}",
                    )
                    plt.tight_layout()
                    plt.show()
                else:
                    print("Permutation importance hâlâ sıfır; özellikler hedefle zayıf ilişkili olabilir.")
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            #### Kavram: permutation importance (permütasyon önemi)

            Bir özelliği rastgele permüte ederek model skorundaki düşüşü ölçer. CV modeli sabit tahmin veriyorsa yeniden eğitimli yorum pipeline’ı (`gbm_permutation_importance`) devreye girer.
                """
            )
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
                    shap.summary_plot(
                        sv, Xs, show=True, max_display=12, plot_size=(16.0, 9.0)
                    )
            except Exception as e:
                print("SHAP atlandı:", e)
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            **SHAP:** Model-agnostic açıklama; örneklem üzerinde yorumlayın, tüm evren için genelleştirmeyin.
                """
            )
        )
    )

    cells.append(md("## 6. Single vs Multi vs Sector-featured (her hedef)"))
    cells.append(
        md(
            section_story(
                bridge="Model tipini seçtik; şimdi **veriyi nasıl havuzlayacağımızı** soruyoruz: tek model mi, ticker başına mı?",
                purpose="Aynı RidgeCV pipeline ile dört mod: pooled zaman sırası, ticker dummy, + sektör getirisi, ticker başına ortalama CV.",
                why_now="Hisse kimliği ve sektör ortak hareketi §2’de gördüğümüz yapıyla uyumlu mu test edilir.",
                when_use="Çok ticker’lı panelde global model vs yerel model trade-off’u raporlanırken.",
            )
            + """

            #### Kavram: havuzlama modları

            - **pooled_by_entry_date:** Tek model, ortak katsayılar; ticker kimliği yok.
            - **multi_ticker_dummies:** One-hot ticker — seviye farkları.
            - **multi_dummies_sector_ret:** Sektör sepet getirisi eklenir.
            - **single_per_ticker_mean_cv_rmse:** Her hisse ayrı model; tabloda ortalama RMSE.
            """
        )
    )
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
            if not tab_multi.empty and viz_target in tab_multi["hedef"].values:
                sub_m = tab_multi[tab_multi["hedef"] == viz_target]
                sub_m.set_index("mode")["cv_rmse_mean"].plot.bar(
                    figsize=(11, 4.5), title=f"Havuzlama modları — {viz_target}"
                )
                plt.ylabel("CV RMSE")
                plt.tight_layout()
                plt.show()
                _bm = sub_m.loc[sub_m["cv_rmse_mean"].idxmin(), "mode"]
                display(
                    Markdown(
                        format_sonuc(
                            "6 Havuzlama",
                            [
                                f"`{viz_target}` için bu veri setinde en iyi havuzlama: **{_bm}**.",
                                "§7’de tüm hedefler için FS + linear + ağaç + havuzlama tek tabloda birleşecek.",
                                "Per-ticker ortalama yüksek `std` ise bazı hisselerde veri az veya gürültülü olabilir.",
                            ],
                        )
                    )
                )
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            - **`tab_multi` / pivot:** Aynı `hedef` için dört `mode` — düşük `cv_rmse_mean` iyi; `best_mode` sütunu kazananı işaret eder.
            - **Pooled vs dummy:** Kimlik bilgisi eklemek bazen hata düşürür, bazen aşırı parametre ve gürültü getirir.
            - **Per-ticker ortalama:** `std` yüksekse hisseler arası tutarsızlık — tek bir global kural yeterli olmayabilir.
                """
            )
        )
    )

    cells.append(md("## 7. Özet: hedef bazında seçilen modeller"))
    cells.append(
        md(
            section_story(
                bridge="§4–§6’da her hedef için ayrı kararlar verdik; şimdi **tek sayfalık hikaye özeti**.",
                purpose="`summary_all`: FS, linear reg, ağaç kazananı, en iyi havuzlama ve `overall_pick` — sınıflandırma sütunlarıyla birlikte.",
                why_now="Rapor ve sunumda “hangi horizon’da ne seçildi?” sorusuna tek tabloyla cevap.",
                when_use="Ödev teslimi, jüri özeti veya kendi notlarınız için son kontrol listesi.",
            )
        )
    )
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
            summary_all["overall_pick"] = summary_all.apply(
                lambda r: (
                    r["best_reg"]
                    if np.isfinite(r.get("cv_tree_best", np.nan))
                    and r["cv_rmse_linear"] <= r["cv_tree_best"]
                    else (r.get("tree_winner") or r["best_reg"])
                ),
                axis=1,
            )
            if "tab_clf_beat" in dir() and not tab_clf_beat.empty:
                summary_all["clf_beat_winner"] = _beat_winner
                summary_all["clf_beat_auc"] = clf_beat_auc
            if "tab_clf_direction" in dir() and not tab_clf_direction.empty:
                summary_all["clf_direction_horizon"] = viz_horizon
                summary_all["clf_direction_winner"] = clf_dir_winner
                summary_all["clf_direction_auc"] = clf_dir_auc
            display(summary_all)
            for _, row in summary_all.iterrows():
                print(
                    f"- {row['hedef']}: FS={row['best_fs']}, linear={row['best_reg']}, "
                    f"ağaç={row.get('tree_winner', '—')}, özet={row['overall_pick']}"
                )
            if "clf_beat_winner" in summary_all.columns:
                print(
                    f"- Sınıflandırma eps_beat: {_beat_winner} (AUC≈{clf_beat_auc:.3f}); "
                    f"yön {viz_horizon}: {clf_dir_winner} (AUC≈{clf_dir_auc:.3f})"
                )
            display(
                Markdown(
                    format_sonuc(
                        "7 Özet",
                        [
                            "Bu tablo ödevin **tek sayfa sonucu**: her `y_*` için hangi FS, linear, ağaç ve havuzlama kazandı.",
                            "`overall_pick`: CV’de linear RMSE ağaçtan iyiyse linear, değilse ağaç — empirik kural, evrensel değil.",
                            "Sınıflandırma sütunları beat/yönü regresyondan ayırır; §8’de satır bazlı tahminlere geçiyoruz.",
                        ],
                    )
                )
            )
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            - **`summary_all`:** Satır = horizon; sütunlar §4–§6 + sınıflandırma özeti.
            - **`overall_pick`:** Rapor cümlesi için kısa etiket; detay için ilgili bölüm tablolarına dönün.
            - **`viz_target`:** En düşük linear CV RMSE — fold ve SHAP grafikleri çoğunlukla burada.
                """
            )
        )
    )

    cells.append(md("## 8. Tahminler: tüm hisseler × tüm hedefler"))
    cells.append(
        md(
            section_story(
                bridge="Özet tabloyu gördük; şimdi **her olay satırı** için seçilen modellerin tahminlerini üretiyoruz.",
                purpose="`predictions_all` (tüm ticker × hedef) ve `pred_rmse_by_ticker`; örnek hedefte holdout vs CV kıyası.",
                why_now="Excel ve sunumda somut sayılar; fakat genelleme yorumu için CV ile ayırmak şart.",
                when_use="Portföy/rapor tablosu istendiğinde; model seçimi için yine §3.5–§7 CV metrikleri esas.",
            )
            + """

            #### Kavram: in-sample vs çapraz doğrulama (CV)

            **In-sample** tahmin: modelin eğitildiği satırlar üzerinde — RMSE genelde **iyimser** (optimistic). **CV RMSE:** gelecek fold’larda ölçülen dürüst skor. Raporlarda CV’yi birincil, bu bölümü ikincil kullanın.
            """
        )
    )
    cells.append(
        code(
            """
            from IPython.display import display

            REG_KEY = {"Ridge": "res_ridge", "Lasso": "res_lasso", "ElasticNet": "res_enet"}


            def _rmse_vec(a, b):
                a = np.asarray(a, dtype=float)
                b = np.asarray(b, dtype=float)
                m = np.isfinite(a) & np.isfinite(b)
                if not m.any():
                    return np.nan
                return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


            pred_parts = []
            for tgt, d in per_target.items():
                X, y, cols, meta = prepare_xy_with_meta(events, tgt, feats)
                Xm = materialize_fs_matrix(d["best_fs"], X, y, d["bundles"])
                rk = REG_KEY[d["best_reg"]]
                yhat_lin = d[rk]["model"].predict(Xm)
                rxd = d.get("rx") or {}
                yhat_xgb = (
                    rxd["model"].predict(Xm)
                    if rxd.get("model") is not None
                    else np.full(len(yhat_lin), np.nan)
                )
                rld = d.get("rl") or {}
                yhat_lgb = (
                    rld["model"].predict(Xm)
                    if rld.get("model") is not None
                    else np.full(len(yhat_lin), np.nan)
                )
                block = meta.copy()
                block["hedef"] = tgt
                block["y_gercek"] = y
                block["y_hat_linear"] = yhat_lin
                block["y_hat_xgb"] = yhat_xgb
                block["y_hat_lgbm"] = yhat_lgb
                block["pred_fs"] = d["best_fs"]
                block["pred_linear_reg"] = d["best_reg"]
                block["linear_residual"] = y - yhat_lin
                pred_parts.append(block)

            predictions_all = pd.concat(pred_parts, ignore_index=True)
            rmse_rows = []
            for (h, tk), g in predictions_all.groupby(["hedef", "ticker"], sort=False):
                rmse_rows.append(
                    {
                        "hedef": h,
                        "ticker": tk,
                        "n": int(len(g)),
                        "rmse_linear": _rmse_vec(g["y_gercek"].values, g["y_hat_linear"].values),
                        "rmse_xgb": _rmse_vec(g["y_gercek"].values, g["y_hat_xgb"].values),
                        "rmse_lgbm": _rmse_vec(g["y_gercek"].values, g["y_hat_lgbm"].values),
                    }
                )
            pred_rmse_by_ticker = pd.DataFrame(rmse_rows)
            print("predictions_all:", predictions_all.shape)
            display(predictions_all.head(25))
            display(pred_rmse_by_ticker.sort_values("rmse_linear").head(25))

            if viz_target in per_target:
                Xh, yh, _ = prepare_xy(events, viz_target, feats)
                Xhm = materialize_fs_matrix(
                    per_target[viz_target]["best_fs"],
                    Xh,
                    yh,
                    per_target[viz_target]["bundles"],
                )
                sp = int(len(Xhm) * 0.8)
                m_h = per_target[viz_target][REG_KEY[per_target[viz_target]["best_reg"]]]["model"]
                m_h.fit(Xhm.iloc[:sp], yh[:sp])
                hold_rmse = _rmse_vec(yh[sp:], m_h.predict(Xhm.iloc[sp:]))
                cv_rm = per_target[viz_target]["cv_best_lin"]
                display(
                    Markdown(
                        format_sonuc(
                            "8 Tahminler",
                            [
                                f"`{viz_target}`: holdout RMSE≈{hold_rmse:.4f}, CV≈{cv_rm:.4f} — yakınsa hikaye tutarlı.",
                                "`predictions_all` raporlama içindir; **model seçimi** için §7 özet ve CV tabloları geçerlidir.",
                                "§9: aynı sonuçları Excel ve sunuma aktarıyoruz.",
                            ],
                        )
                    )
                )
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            - **`predictions_all`:** `y_gercek`, `y_hat_*`, `pred_fs`, `pred_linear_reg` — hangi pipeline kullanıldığını gösterir.
            - **`pred_rmse_by_ticker`:** Hangi (ticker, hedef) çiftinde hata şişiyor — in-sample RMSE, dikkatli yorum.
                """
            )
        )
    )

    cells.append(md("## 9. Teslim dosyaları (Excel + PPTX)"))
    cells.append(
        md(
            section_story(
                bridge="Analiz bitti; son adım **raporlama (delivery)** — öğretmen ve jüri için dosyalar.",
                purpose="`data_*.xlsx` (panel, korelasyon, tahminler, sınıflandırma özetleri) ve `presentation_*.pptx` iskeleti.",
                why_now="Ödev formatı Excel + sunum istiyor; not defteri ile dosyaların uyumlu olması gerekir.",
                when_use="Teslim öncesi `STUDENT_SURNAME` güncelleyip tüm hücreleri baştan çalıştırın.",
            )
        )
    )
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
            _clf_beat_x = tab_clf_beat if "tab_clf_beat" in dir() else None
            _clf_dir_x = tab_clf_direction if "tab_clf_direction" in dir() else None
            export_excel(
                xlsx_path,
                adj,
                events,
                corr=corr_primary,
                corr_by_window=corr_sheets,
                predictions=predictions_all,
                pred_rmse_by_ticker=pred_rmse_by_ticker,
                clf_beat_summary=_clf_beat_x,
                clf_direction_summary=_clf_dir_x,
            )
            print("Yazıldı:", xlsx_path)

            bullets = [
                ("Veri ve hedef", ["yfinance panel", "7 işgünü horizon", "Özellikler giriş öncesi"]),
                ("Korelasyon", ["1–24 ay takvim pencereleri", "Isı haritası + 1y/2y clustermap", "NVDA–INTC rolling"]),
                (
                    "Modeller",
                    [
                        "Linear: FS + Ridge/Lasso/ENet",
                        "LDA / QDA / Gaussian NB (zaman serisi CV, ROC-AUC)",
                        "XGB vs LGBM RandomizedSearch",
                    ],
                ),
                ("Sonuç", ["Hedef bazında FS+reg seçimi", "viz_target grafikleri", "GBM + multi özet"]),
            ]
            pred_lines = [
                f"Tahmin satırı (tüm hedef × olay): {len(predictions_all)}",
                "Excel: predictions_all + pred_rmse_by_ticker sayfaları",
                "Özet: in-sample tahmin; genelleme için CV RMSE tablolarına bakın.",
            ]
            if len(pred_rmse_by_ticker):
                worst = pred_rmse_by_ticker.nlargest(3, "rmse_linear")
                pred_lines.append(
                    "Yüksek linear RMSE (örnek): "
                    + ", ".join(
                        f"{r.ticker}/{r.hedef}={r.rmse_linear:.4f}"
                        for _, r in worst.iterrows()
                    )
                )
            bullets.append(("Tahminler (tüm hisseler)", pred_lines))
            pptx_path = f"presentation_{STUDENT_SURNAME}.pptx"
            export_presentation(
                pptx_path,
                title="Earnings sonrası getiri tahmini",
                bullets=bullets,
            )
            print("Yazıldı:", pptx_path)
            display(
                Markdown(
                    format_sonuc(
                        "9 Teslim",
                        [
                            f"Dosyalar hazır: `{xlsx_path}`, `{pptx_path}` — soyadınızı kontrol edin.",
                            "Excel’deki tahminler in-sample; slaytlarda **CV RMSE ve kazanan model** cümlelerini §7’den alın.",
                            "Hikaye tamam: veri → CV → modeller → özet → tahmin → teslim.",
                        ],
                    )
                )
            )
            """
        )
    )

    cells.append(
        md(
            section_reading(
                """
            - **Excel:** `events_features` ana panel; `predictions_all` / `pred_rmse_by_ticker` §8 çıktısı; korelasyon sayfaları §2.
            - **PPTX:** Akış özeti; rakamları §7 `summary_all` ve CV tablolarından elle doğrulayın.
                """
            )
        )
    )

    cells.append(
        md(
            """
            ## Checklist (ödev tablosu)

            - [x] Anlatı akışı: her § için “Bu adımda ne yapıyoruz?” + çıktı okuma + Sonuç
            - [x] Korelasyon matrisi + heatmap / clustermap / rolling
            - [x] Zaman serisi CV (blok fold, gap/offset) — tüm model kıyasları
            - [x] Regresyon: FS + Ridge/Lasso/ElasticNet + learning curve
            - [x] Sınıflandırma: LDA / QDA / Naive Bayes (`eps_beat` + getiri yönü, ROC-AUC)
            - [x] XGB vs LGBM + permutation / SHAP
            - [x] Havuzlama: pooled / dummy / sektör / per-ticker
            - [x] Hedef bazında özet + tahmin tablosu + Excel + PPTX
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
