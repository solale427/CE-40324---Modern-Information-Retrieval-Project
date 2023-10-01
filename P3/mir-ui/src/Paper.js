import React, {useState} from 'react';

const SearchPage = () => {
    const [title_query, setSearchTerm1] = useState('');
    const [abstract_query, setSearchTerm2] = useState('');
    const [weight, setSearchTerm3] = useState('');
    const [searchResults, setSearchResults] = useState([]);

    const handleSearch = async (e) => {
        e.preventDefault()
        try {
            const response = await fetch('http://localhost:5000/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    title_query,
                    abstract_query,
                    weight,
                }),
            });

            if (response.ok) {
                const data = await response.json();
                setSearchResults(data.results);
            } else {
                console.error('API error:', response.statusText);
            }
        } catch (error) {
            console.error('Network error:', error);
        }
    };


    return (
        <div style={styles.container}>
            <h1 style={styles.title}>Search Page</h1>
            <form style={styles.searchContainer} onSubmit={handleSearch}>
                <input
                    type="text"
                    value={title_query}
                    onChange={(e) => setSearchTerm1(e.target.value)}
                    placeholder="Title Query"
                    style={styles.input}
                />
                <input
                    type="text"
                    value={abstract_query}
                    onChange={(e) => setSearchTerm2(e.target.value)}
                    placeholder="Abstract Query"
                    style={styles.input}
                />
                <input
                    type="text"
                    value={weight}
                    onChange={(e) => setSearchTerm3(e.target.value)}
                    placeholder="Title Weight"
                    style={styles.input}
                />
                <button style={styles.button} type="submit">Search</button>
            </form>

            <h2 style={styles.resultsTitle}>Search Results</h2>
            {searchResults.length === 0 ? (
                <p style={styles.noResults}>No results found.</p>
            ) : (
                searchResults.map((result, index) => (
                    <div key={index} style={styles.resultContainer}>
                        <div style={styles.line}></div>
                        <a href={result.url}><h3 style={styles.resultTitle}>{result.title}</h3></a>
                        <p style={styles.resultAbstract}>{result.abstract}</p>
                    </div>
                ))
            )}
        </div>
    );
};

const styles = {
    container: {
        maxWidth: '600px',
        margin: '0 auto',
        padding: '20px',
        fontFamily: 'Arial, sans-serif',
    },
    title: {
        fontSize: '24px',
        marginBottom: '20px',
    },
    searchContainer: {
        display: 'flex',
        marginBottom: '20px',
    },
    input: {
        marginRight: '10px',
        padding: '5px',
        fontSize: '16px',
        borderRadius: '4px',
        border: '1px solid #ccc',
        flex: 1,
    },
    button: {
        padding: '5px 10px',
        fontSize: '16px',
        borderRadius: '4px',
        background: '#007bff',
        color: '#fff',
        border: 'none',
        cursor: 'pointer',
    },
    resultsTitle: {
        fontSize: '20px',
        marginBottom: '10px',
    },
    noResults: {
        fontStyle: 'italic',
        color: '#888',
    },
    resultContainer: {
        marginBottom: '20px',
    },
    resultTitle: {
        fontSize: '18px',
        marginBottom: '5px',
    },
    resultAbstract: {
        fontSize: '14px',
    },
    line: {
        borderBottom: '1px solid black'
    },
};

export default SearchPage;
