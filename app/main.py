# app/main.py

import streamlit as st

try:
    from backend.llm_query import parse_query_with_mistral
    from backend.recommender import personalized_search, rerank_results

    st.title("📚 Smart Book Recommender")

    # User selection (for personalization)
    user_id = st.selectbox("Choose user:", ["user_1", "guest"])

    # User query input
    user_query = st.text_input("What book are you looking for?")

    if user_query:
        # Step 1: Refine query using LLM
        with st.spinner("Understanding your query with Mistral..."):
            refined_query = parse_query_with_mistral(user_query)
            st.success("✅ Query refined")
            st.write(f"*Refined Query:* {refined_query}")

        # Step 2: Search vector DB with personalization
        with st.spinner("Searching books..."):
            results = personalized_search(refined_query, user_id)

        # Step 3: Rerank results using LLM
        with st.spinner("Reranking results..."):
            ranked_titles = list(rerank_results(refined_query, results))

        # Step 4: Sort by reranked order
        ranked_items = sorted(
            results,
            key=lambda x: ranked_titles.index(x['metadata']['title'])
            if x['metadata']['title'] in ranked_titles else 999
        )

        # Step 5: Display books
        st.subheader("📖 Recommended Books")
        for item in ranked_items:
            st.markdown(f"### {item['metadata']['title']} by {item['metadata']['author']}")
            st.write(item['metadata']['description'])

except Exception as e:
    st.error(f"⚠ An error occurred:\n\n{e}")
    