from pathlib import Path

import streamlit as st

from engine.ai_refinement import DEFAULT_OLLAMA_MODEL, refine_transactions
# from engine.huggingface_refinement import (
#     DEFAULT_HUGGINGFACE_MODEL,
#     HUGGINGFACE_MODEL_OPTIONS,
#     refine_transactions_with_huggingface,
# )
from engine.loader import load_transactions
from engine.pipeline import process_transactions
from engine.taxonomy import APPROVED_CATEGORIES
from engine.training_data import (
    DEFAULT_TRAINING_DATASET_PATH,
    available_review_categories,
    build_training_record,
    load_review_index,
    load_review_records,
    records_to_jsonl,
    select_review_candidates,
    training_corpus_stats,
    upsert_review_records,
)


AI_REFINEMENT_COLUMNS = [
    "AI Decision",
    "AI Suggested Category",
    "Refinement Type",
    "AI Semantic Reason",
    "AI Rule Suggestion",
    "AI Missing Signal",
    "AI Confidence Advisory",
    "Validation Status",
]


def _ensure_columns(df, columns):
    for column in columns:
        if column not in df.columns:
            df[column] = ""

    return df


def _is_blank(value):
    if value is None:
        return True

    try:
        if value != value:
            return True
    except (TypeError, ValueError):
        pass

    return str(value).strip() == ""


def _amount_value(row):
    amount = row.get("Amount", "")

    if not _is_blank(amount):
        return amount

    direction = row.get("Direction", "")
    debit = row.get("Debits", "")
    credit = row.get("Credits", "")

    if direction == "OUT" and not _is_blank(debit):
        return debit

    if direction == "IN" and not _is_blank(credit):
        return credit

    if not _is_blank(debit):
        return debit

    if not _is_blank(credit):
        return credit

    return ""


def _source_row_index(index_value):
    try:
        return int(index_value)
    except (TypeError, ValueError):
        return str(index_value)


def _attach_ai_refinement_results(processed_df):
    review_df = processed_df.copy()
    refinement_results = st.session_state.get("ai_refinement_results", [])

    _ensure_columns(review_df, AI_REFINEMENT_COLUMNS)

    for result in refinement_results:
        row_index = result.get("Row Index")

        if row_index not in review_df.index:
            continue

        for column in AI_REFINEMENT_COLUMNS:
            review_df.at[row_index, column] = result.get(column, "")

    return review_df


def _render_classification_output(raw_df, processed_df):
    st.subheader("Raw Transactions")
    st.dataframe(raw_df, use_container_width=True)

    st.subheader("Transaction Parsing Output")

    parser_columns = [
        "Normalized Narration",
        "Transaction Prefix",
        "Transaction Subtype",
        "Protocol Family",
        "Reference ID",
        "Entity Name",
        "Bank Name",
        "UPI ID",
        "UPI Handle",
        "Parser Rule",
        "Parser Confidence",
    ]

    st.dataframe(
        processed_df[
            [
                column
                for column in parser_columns
                if column in processed_df.columns
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Classification Output")

    classification_columns = [
        "Bank",
        "Narration",
        "Normalized Narration",
        "Direction",
        "Mode",
        "Entity Name",
        "UPI ID",
        "UPI Handle",
        "Parse Quality",
        "Protocol Family",
        "Parser Rule",
        "Parser Confidence",
        "Instrument Type",
        "Intent Tags",
        "Movement Tags",
        "Bank Family",
        "Merchant",
        "Bounce Flag",
        "Charge Flag",
        "Reversal Flag",
        "Salary Flag",
        "Tax Flag",
        "Cash Flag",
        "Deposit Flag",
        "Withdrawal Flag",
        "ATM Flag",
        "Cheque Flag",
        "Investment Flag",
        "Insurance Flag",
        "Recharge Flag",
        "Travel Flag",
        "Utility Flag",
        "Loan Flag",
        "Entity Type",
        "Entity Confidence",
        "Transaction Prefix",
        "Transaction Subtype",
        "Reference ID",
        "Bank Name",
        "Confidence",
        "Decision Path",
        "Conflicts",
        "Ranked Candidates",
        "Alternative Categories",
        "Review Required",
        "Review Reason",
        "Evidence Summary",
        "Category",
        "Matched Rule",
    ]

    if "Old Category" in processed_df.columns:
        category_index = classification_columns.index("Category")
        classification_columns.insert(category_index + 1, "Old Category")

    st.dataframe(
        processed_df[
            [
                column
                for column in classification_columns
                if column in processed_df.columns
            ]
        ],
        use_container_width=True,
    )


def _render_training_review(processed_df):
    st.subheader("Training Review")

    dataset_path = st.text_input(
        "Training corpus path",
        value=DEFAULT_TRAINING_DATASET_PATH,
        key="training_dataset_path",
    )

    try:
        review_records = load_review_records(dataset_path)
        review_index = load_review_index(dataset_path)
    except ValueError as exc:
        st.error(str(exc))
        return

    stats = training_corpus_stats(review_records)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Reviewed", stats["total"])
    metric_cols[1].metric("Corrections", stats["corrections"])
    metric_cols[2].metric("Accepted", stats["accepted_deterministic"])
    metric_cols[3].metric("Categories", len(stats["category_counts"]))

    filter_cols = st.columns(4)
    max_rows = filter_cols[0].number_input(
        "Queue size",
        min_value=1,
        max_value=1000,
        value=50,
        step=10,
        key="training_max_rows",
    )
    max_confidence = filter_cols[1].slider(
        "Max confidence",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.01,
        key="training_max_confidence",
    )
    min_priority = filter_cols[2].slider(
        "Min priority",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key="training_min_priority",
    )
    feedback_source = filter_cols[3].text_input(
        "Feedback source",
        value="streamlit_training_review",
        key="training_feedback_source",
    )

    option_cols = st.columns(6)
    hide_generic_high_confidence = option_cols[0].checkbox(
        "Hide generic EFT",
        value=True,
        key="training_hide_generic_eft",
    )
    review_required_only = option_cols[1].checkbox(
        "Review required",
        value=False,
        key="training_review_required_only",
    )
    conflict_only = option_cols[2].checkbox(
        "Conflicts only",
        value=False,
        key="training_conflict_only",
    )
    include_reviewed = option_cols[3].checkbox(
        "Show reviewed",
        value=False,
        key="training_include_reviewed",
    )
    allow_updates = option_cols[4].checkbox(
        "Update existing",
        value=False,
        key="training_allow_updates",
    )
    show_review_details = option_cols[5].checkbox(
        "Show details",
        value=False,
        key="training_show_review_details",
    )

    max_confidence_filter = None

    if max_confidence < 1.0:
        max_confidence_filter = max_confidence

    available_category_options = available_review_categories(
        processed_df,
        review_index=review_index,
        include_reviewed=include_reviewed,
        hide_generic_high_confidence=hide_generic_high_confidence,
        review_required_only=review_required_only,
        conflict_only=conflict_only,
        max_confidence=max_confidence_filter,
        min_priority=min_priority,
    )
    current_category_filter = st.session_state.get("training_category_filter", [])
    selected_category_filter = [
        category
        for category in current_category_filter
        if category in available_category_options
    ]

    st.session_state["training_category_filter"] = selected_category_filter

    category_filter = st.multiselect(
        "Category filter",
        options=available_category_options,
        key="training_category_filter",
    )
    reviewer = st.text_input(
        "Reviewer",
        value="",
        key="training_reviewer",
    )

    candidates = select_review_candidates(
        processed_df,
        review_index=review_index,
        include_reviewed=include_reviewed,
        hide_generic_high_confidence=hide_generic_high_confidence,
        review_required_only=review_required_only,
        conflict_only=conflict_only,
        category_filter=category_filter,
        max_confidence=max_confidence_filter,
        min_priority=min_priority,
        max_rows=int(max_rows),
    )

    if candidates.empty:
        st.info("No transactions match the current review queue filters.")
    else:
        editor_df = candidates.copy()
        editor_df.insert(0, "Submit", False)
        editor_df.insert(1, "Source Row", editor_df.index)
        _ensure_columns(
            editor_df,
            [
                "Old Category",
                "Category",
                "Amount",
                "Debits",
                "Credits",
                "Direction",
                "Confidence",
            ],
        )
        editor_df["New Category"] = editor_df["Category"]
        editor_df["Amount"] = [
            _amount_value(row)
            for _, row in editor_df.iterrows()
        ]
        editor_df["Existing Verified Category"] = editor_df["Transaction ID"].map(
            lambda transaction_id: review_index.get(
                transaction_id,
                {},
            ).get("correct_category", "")
        )
        editor_df["Verified Category"] = editor_df.apply(
            lambda row: row["Existing Verified Category"] or row.get("Category", ""),
            axis=1,
        )
        editor_df["Feedback Note"] = ""

        priority_review_columns = [
            "Submit",
            "Verified Category",
            "Old Category",
            "New Category",
            "Amount",
            "Narration",
            "Direction",
            "Confidence",
        ]
        detail_review_columns = [
            "Bank",
            "Bank Name",
            "Review Priority",
            "Priority Flags",
            "Already Reviewed",
            "Existing Verified Category",
            "Entity Name",
            "Merchant",
            "Entity Type",
            "Entity Confidence",
            "Mode",
            "Protocol Family",
            "Transaction Subtype",
            "Conflicts",
            "Ranked Candidates",
            "AI Suggested Category",
            "AI Decision",
            "Review Reason",
            "Source Row",
            "Transaction ID",
            "Feedback Note",
        ]
        review_columns = priority_review_columns

        if show_review_details:
            review_columns = priority_review_columns + detail_review_columns

        _ensure_columns(editor_df, review_columns)

        editable_columns = {
            "Submit",
            "Verified Category",
            "Feedback Note",
        }
        disabled_columns = [
            column
            for column in review_columns
            if column not in editable_columns
        ]

        edited_reviews = st.data_editor(
            editor_df[review_columns],
            key="training_review_editor",
            use_container_width=True,
            hide_index=True,
            height=620,
            disabled=disabled_columns,
            column_config={
                "Submit": st.column_config.CheckboxColumn(
                    "Submit",
                    width="small",
                ),
                "Verified Category": st.column_config.SelectboxColumn(
                    "Verified Category",
                    options=APPROVED_CATEGORIES,
                    required=True,
                    width="medium",
                ),
                "Confidence": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.2f",
                    width="small",
                ),
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    format="%.2f",
                    width="small",
                ),
                "Old Category": st.column_config.TextColumn(
                    "Old Category",
                    width="medium",
                ),
                "New Category": st.column_config.TextColumn(
                    "New Category",
                    width="medium",
                ),
                "Direction": st.column_config.TextColumn(
                    "Direction",
                    width="small",
                ),
                "Review Priority": st.column_config.NumberColumn(
                    "Priority",
                    format="%.1f",
                    width="small",
                ),
                "Already Reviewed": st.column_config.CheckboxColumn(
                    "Reviewed",
                    width="small",
                ),
                "Transaction ID": st.column_config.TextColumn(
                    "Transaction ID",
                    width="medium",
                ),
                "Feedback Note": st.column_config.TextColumn(
                    "Note",
                    width="medium",
                ),
            },
        )

        selected_reviews = edited_reviews[
            edited_reviews["Submit"] == True
        ]

        if st.button(
            "Save Reviews",
            key="training_save_reviews",
            type="primary",
        ):
            if selected_reviews.empty:
                st.warning("Select at least one review row to save.")
            else:
                category_counts = processed_df["Category"].value_counts().to_dict()
                records_to_save = []

                try:
                    for source_index, edited_row in selected_reviews.iterrows():
                        source_row = candidates.loc[source_index]

                        records_to_save.append(
                            build_training_record(
                                source_row,
                                final_category=edited_row["Verified Category"],
                                feedback_source=feedback_source,
                                reviewer=reviewer,
                                feedback_note=edited_row.get("Feedback Note", ""),
                                row_index=_source_row_index(source_row.name),
                                category_counts=category_counts,
                            )
                        )

                    save_stats = upsert_review_records(
                        records_to_save,
                        path=dataset_path,
                        allow_updates=allow_updates,
                    )
                    st.success(
                        "Saved "
                        f"{save_stats['inserted']} new and "
                        f"{save_stats['updated']} updated reviews."
                    )

                    if save_stats["skipped_duplicates"]:
                        st.warning(
                            f"Skipped {save_stats['skipped_duplicates']} already-reviewed rows. "
                            "Enable Update existing to replace them."
                        )
                except ValueError as exc:
                    st.error(str(exc))

    if review_records:
        st.subheader("Training Corpus")
        category_rows = [
            {
                "Category": category,
                "Reviewed Rows": count,
            }
            for category, count in sorted(
                stats["category_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        st.dataframe(
            category_rows,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download JSONL",
            data=records_to_jsonl(review_records) + "\n",
            file_name=Path(dataset_path).name or "training_reviews.jsonl",
            mime="application/jsonl",
            key="training_download_jsonl",
        )


def _render_ai_refinement(processed_df):
    st.subheader("AI Refinement")

    refinement_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.01,
    )

    refinement_model = st.text_input(
        "Ollama model",
        value=DEFAULT_OLLAMA_MODEL,
    )

    refinement_max_rows = st.number_input(
        "Max rows to refine",
        min_value=1,
        max_value=500,
        value=25,
        step=1,
    )

    include_old_category_disagreement = st.checkbox(
        "Include old-vs-new category disagreements",
        value=True,
    )

    if st.button("AI Refinement"):
        with st.spinner("Running advisory AI refinement..."):
            refinement_results = refine_transactions(
                processed_df,
                threshold=refinement_threshold,
                model=refinement_model,
                max_rows=int(refinement_max_rows),
                include_old_category_disagreement=include_old_category_disagreement,
            )

        st.session_state["ai_refinement_results"] = refinement_results

    refinement_results = st.session_state.get("ai_refinement_results", [])

    if refinement_results:
        st.dataframe(refinement_results, use_container_width=True)
    else:
        st.info("No AI refinement results are available for this run.")

    # =========================
    # HUGGING FACE AI REFINEMENT
    # =========================
    #
    # st.subheader("Hugging Face AI Refinement")
    #
    # hf_refinement_threshold = st.slider(
    #     "Low-confidence threshold",
    #     min_value=0.0,
    #     max_value=1.0,
    #     value=0.65,
    #     step=0.01,
    #     key="hf_refinement_threshold"
    # )
    #
    # hf_model_preset = st.selectbox(
    #     "Hugging Face model route",
    #     options=HUGGINGFACE_MODEL_OPTIONS + ["Custom"],
    #     index=HUGGINGFACE_MODEL_OPTIONS.index(DEFAULT_HUGGINGFACE_MODEL),
    #     key="hf_model_preset"
    # )
    #
    # hf_custom_model = st.text_input(
    #     "Custom Hugging Face model route",
    #     value="",
    #     disabled=hf_model_preset != "Custom",
    #     key="hf_custom_model"
    # )
    # st.caption(
    #     "Use a model route supported by an enabled provider. If Custom is selected, paste the exact string from the Hugging Face Playground."
    # )
    #
    # hf_refinement_model = (
    #     hf_custom_model.strip()
    #     if hf_model_preset == "Custom" and hf_custom_model.strip()
    #     else hf_model_preset
    # )
    #
    # hf_access_token = st.text_input(
    #     "Hugging Face access token",
    #     type="password",
    #     key="hf_access_token"
    # )
    #
    # hf_refinement_max_rows = st.number_input(
    #     "Max rows to refine",
    #     min_value=1,
    #     max_value=500,
    #     value=25,
    #     step=1,
    #     key="hf_refinement_max_rows"
    # )
    #
    # hf_include_old_category_disagreement = st.checkbox(
    #     "Include old-vs-new category disagreements",
    #     value=False,
    #     key="hf_include_old_category_disagreement"
    # )
    #
    # hf_request_json_response = st.checkbox(
    #     "Request provider JSON mode",
    #     value=False,
    #     key="hf_request_json_response"
    # )
    # st.caption(
    #     "Leave this off if the hosted provider returns 400 for response_format. The local validator still extracts and validates JSON."
    # )
    #
    # if st.button(
    #     "Hugging Face Refinement",
    #     key="hf_refinement_button"
    # ):
    #
    #     with st.spinner("Running Hugging Face refinement..."):
    #
    #         hf_refinement_results = refine_transactions_with_huggingface(
    #             processed_df,
    #             threshold=hf_refinement_threshold,
    #             model=hf_refinement_model,
    #             token=hf_access_token.strip() or None,
    #             max_rows=int(hf_refinement_max_rows),
    #             include_old_category_disagreement=hf_include_old_category_disagreement,
    #             request_json_response=hf_request_json_response
    #         )
    #
    #     if hf_refinement_results:
    #
    #         st.dataframe(
    #             hf_refinement_results
    #         )
    #
    #     else:
    #
    #         st.info(
    #             "No rows matched the Hugging Face refinement routing criteria."
    #         )


st.set_page_config(
    page_title="Categorization Engine",
    layout="wide",
)

st.title("Transaction Categorization Engine")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"],
)

sheet_name = st.text_input(
    "Enter Sheet Name",
    value="Xns Transactions",
)

if uploaded_file and sheet_name:
    try:
        raw_df = load_transactions(
            uploaded_file,
            sheet_name,
        )
        processed_df = process_transactions(raw_df.copy())
        review_df = _attach_ai_refinement_results(processed_df)

        st.success("File Loaded Successfully")

        output_tab, training_tab, ai_tab, ml_tab = st.tabs(
            [
                "Classification Output",
                "Training Review",
                "AI Refinement",
                "ML Evaluation",
            ]
        )

        with output_tab:
            _render_classification_output(raw_df, processed_df)

        with training_tab:
            _render_training_review(review_df)

        with ai_tab:
            _render_ai_refinement(processed_df)

        with ml_tab:
            st.subheader("ML Model Evaluation")
            st.write("Load a trained TF-IDF + LogisticRegression pipeline and evaluate it against the uploaded transactions. This compares the rule-based Category to the ML prediction when an 'Old Category' (label) is present.")

            # Model selection
            model_type = st.radio(
                "Select ML Model",
                options=["Basic (TF-IDF only)", "Enhanced (TF-IDF + Context: salary, recurring)"],
                help="Basic: narration-only TF-IDF | Enhanced: adds is_recurring and salary_probability features",
                horizontal=True
            )

            if model_type == "Basic (TF-IDF only)":
                model_path = st.text_input("Model path", value=r"models/tfidf_logreg_pipeline.pkl")
                is_context_aware = False
            else:
                model_path = st.text_input("Model path", value=r"models/enhanced_model_with_context.pkl")
                is_context_aware = True

            decision_mode = st.selectbox(
                "Decision Mode",
                options=["ML-only", "Rule-only", "Hybrid (rule overrides when Parser Confidence >= threshold)"],
                index=0,
            )

            hybrid_threshold = None
            if decision_mode.startswith("Hybrid"):
                hybrid_threshold = st.slider("Parser Confidence threshold (rule override)", min_value=0.0, max_value=1.0, value=0.8, step=0.01)

            gen_report = st.checkbox("Generate detailed error report (confusion matrix, per-class metrics, mismatches CSV)", value=True)

            run_button = st.button("Run ML Evaluation")

            if run_button:
                try:
                    import pickle, json, io
                    import pandas as pd
                    import numpy as np
                    from pathlib import Path
                    try:
                        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
                    except Exception as exc:
                        st.error(f"scikit-learn is required for ML evaluation: {exc}")
                        raise

                    model_file = Path(model_path)
                    if not model_file.exists():
                        st.error(f"Model file not found: {model_path}")
                    else:
                        with open(model_file, 'rb') as f:
                            model_artifact = pickle.load(f)

                        if 'Normalized Narration' not in processed_df.columns:
                            st.error("Processed data missing 'Normalized Narration' column.")
                        else:
                            texts = processed_df['Normalized Narration'].astype(str).to_list()

                            # Extract model components based on type
                            if is_context_aware:
                                # Enhanced model: extract TF-IDF, scaler, clf from artifact dict
                                try:
                                    tfidf = model_artifact.get('tfidf')
                                    scaler = model_artifact.get('scaler')
                                    clf = model_artifact.get('clf')

                                    if not all([tfidf, scaler, clf]):
                                        st.error("Invalid enhanced model artifact structure.")
                                    else:
                                        # Transform narrations with TF-IDF
                                        X_tfidf = tfidf.transform(texts)

                                        # Add contextual features
                                        is_recurring_arr = processed_df.get('is_recurring', pd.Series([0]*len(texts))).astype(int).values
                                        salary_prob_arr = processed_df.get('salary_probability', pd.Series([0.0]*len(texts))).astype(float).values

                                        # If missing, warn user
                                        if 'is_recurring' not in processed_df.columns:
                                            st.warning("is_recurring column not found; using zeros. For best results, regenerate features with train_enhanced_with_context.py")
                                        if 'salary_probability' not in processed_df.columns:
                                            st.warning("salary_probability column not found; using zeros.")

                                        X_contextual = np.column_stack([is_recurring_arr, salary_prob_arr])
                                        X_contextual = scaler.transform(X_contextual)

                                        from scipy.sparse import hstack
                                        X_combined = hstack([X_tfidf, X_contextual])

                                        preds = clf.predict(X_combined)
                                        try:
                                            prob_arr = clf.predict_proba(X_combined)
                                            max_probs = prob_arr.max(axis=1)
                                        except Exception:
                                            max_probs = [None] * len(preds)

                                except Exception as e:
                                    st.error(f"Failed to load enhanced model: {e}")
                                    raise

                            else:
                                # Basic model: pipeline with fit/predict
                                pipeline = model_artifact
                                preds = pipeline.predict(texts)
                                try:
                                    prob_arr = pipeline.predict_proba(texts)
                                    max_probs = prob_arr.max(axis=1)
                                except Exception:
                                    max_probs = [None] * len(preds)

                            ml_results = processed_df.copy()
                            ml_results['ML Prediction'] = preds
                            ml_results['ML Confidence'] = max_probs
                            ml_results['Model Type'] = model_type

                            # Compute combined decision if requested
                            if decision_mode == 'ML-only':
                                ml_results['Final Category'] = ml_results['ML Prediction']
                            elif decision_mode == 'Rule-only':
                                ml_results['Final Category'] = ml_results['Category']
                            else:  # Hybrid
                                # use rule when Parser Confidence >= threshold, else ML
                                def choose_final(row):
                                    try:
                                        pc = float(row.get('Parser Confidence') or 0.0)
                                    except Exception:
                                        pc = 0.0
                                    if pc >= hybrid_threshold:
                                        return row.get('Category')
                                    return row.get('ML Prediction')

                                ml_results['Final Category'] = ml_results.apply(choose_final, axis=1)

                            st.markdown("### Example predictions (first 50)")
                            display_cols = [c for c in [
                                'Narration', 'Normalized Narration', 'Old Category', 'Category', 'Parser Confidence', 'ML Prediction', 'ML Confidence', 'Final Category'
                            ] if c in ml_results.columns]
                            st.dataframe(ml_results[display_cols].head(50), use_container_width=True)

                            if 'Old Category' in ml_results.columns and gen_report:
                                y_true = ml_results['Old Category'].astype(str).to_list()
                                y_pred_ml = ml_results['ML Prediction'].astype(str).to_list()
                                y_pred_final = ml_results['Final Category'].astype(str).to_list()

                                # ML-only metrics
                                acc_ml = accuracy_score(y_true, y_pred_ml)
                                prec_ml, rec_ml, f1_ml, _ = precision_recall_fscore_support(y_true, y_pred_ml, average='weighted', zero_division=0)

                                # Final metrics (depending on decision mode)
                                acc_final = accuracy_score(y_true, y_pred_final)
                                prec_final, rec_final, f1_final, _ = precision_recall_fscore_support(y_true, y_pred_final, average='weighted', zero_division=0)

                                metric_cols = st.columns(6)
                                metric_cols[0].metric(f'{model_type.split("(")[0].strip()} Accuracy', f"{acc_ml:.4f}")
                                metric_cols[1].metric(f'{model_type.split("(")[0].strip()} Precision', f"{prec_ml:.4f}")
                                metric_cols[2].metric(f'{model_type.split("(")[0].strip()} Recall', f"{rec_ml:.4f}")
                                metric_cols[3].metric(f'{model_type.split("(")[0].strip()} F1 (w)', f"{f1_ml:.4f}")
                                metric_cols[4].metric('Final Accuracy', f"{acc_final:.4f}")
                                metric_cols[5].metric('Final F1 (w)', f"{f1_final:.4f}")

                                # Per-class metrics
                                labels_unique = sorted(list(set(y_true)))
                                per_class = []
                                p, r, f, s = precision_recall_fscore_support(y_true, y_pred_final, labels=labels_unique, zero_division=0)
                                for lab, pp, rr, ff, supp in zip(labels_unique, p, r, f, s):
                                    per_class.append({'label': lab, 'precision': float(pp), 'recall': float(rr), 'f1': float(ff), 'support': int(supp)})

                                per_class_df = None
                                try:
                                    per_class_df = pd.DataFrame(per_class)
                                    st.markdown('### Per-class metrics (Final)')
                                    st.dataframe(per_class_df, use_container_width=True)
                                except Exception:
                                    pass

                                # Confusion matrix for final predictions
                                cm = confusion_matrix(y_true, y_pred_final, labels=labels_unique)
                                cm_df = None
                                try:
                                    cm_df = pd.DataFrame(cm, index=labels_unique, columns=labels_unique)
                                    st.markdown('### Confusion Matrix (Final)')
                                    st.dataframe(cm_df, use_container_width=True)
                                except Exception:
                                    pass

                                # Top mismatches with examples
                                mismatches = ml_results[ml_results['Old Category'].astype(str) != ml_results['Final Category'].astype(str)].copy()
                                mismatches['Old Category'] = mismatches['Old Category'].astype(str)
                                mismatches['Final Category'] = mismatches['Final Category'].astype(str)
                                mismatches['ML Prediction'] = mismatches['ML Prediction'].astype(str)

                                st.markdown(f"### Mismatches (count: {len(mismatches)})")
                                if not mismatches.empty:
                                    mm_display_cols = [c for c in [
                                        'Narration', 'Normalized Narration', 'Old Category', 'Category', 'Parser Confidence', 'ML Prediction', 'ML Confidence', 'Final Category'
                                    ] if c in mismatches.columns]
                                    st.dataframe(mismatches[mm_display_cols].head(200), use_container_width=True)

                                # Provide downloads for detailed reports
                                if per_class_df is not None:
                                    buf = io.StringIO()
                                    per_class_df.to_csv(buf, index=False)
                                    st.download_button('Download per-class metrics CSV', data=buf.getvalue(), file_name='per_class_metrics.csv')

                                if cm_df is not None:
                                    buf2 = io.StringIO()
                                    cm_df.to_csv(buf2)
                                    st.download_button('Download confusion matrix CSV', data=buf2.getvalue(), file_name='confusion_matrix.csv')

                                if not mismatches.empty:
                                    buf3 = io.StringIO()
                                    mismatches.to_csv(buf3, index=False)
                                    st.download_button('Download mismatches CSV', data=buf3.getvalue(), file_name='mismatches.csv')

                            else:
                                st.info('Old Category labels are not present in the uploaded file or detailed report not requested.')

                except Exception as e:
                    st.error(f"ML evaluation failed: {e}")

    except Exception as e:
        st.error(f"Error: {e}")
