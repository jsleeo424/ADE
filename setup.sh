mkdir -p ~/.streamlit/
	
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = 3000\n\
" > ~/.streamlit/config.toml
