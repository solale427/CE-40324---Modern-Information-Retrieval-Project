import React, { useEffect, useState } from 'react';

const styles = {
  container: {
    fontFamily: 'Arial, sans-serif',
    maxWidth: '600px',
    margin: '0 auto',
    padding: '20px',
  },
  header: {
    fontSize: '24px',
    marginBottom: '10px',
  },
  list: {
    marginBottom: '20px',
    padding: '0',
    listStyle: 'none',
  },
  listItem: {
    marginBottom: '5px',
    padding: '5px',
    borderRadius: '5px',
    backgroundColor: '#f2f2f2',
  },
  title: {
    fontSize: '18px',
    fontWeight: 'bold',
    marginBottom: '5px',
  },
  score: {
    fontSize: '14px',
    color: '#666666',
  },
};

const AuthorList = () => {
  const [topHubs, setTopHubs] = useState([]);
  const [topAuthorities, setTopAuthorities] = useState([]);

  useEffect(() => {
    fetchAuthors();
  }, []);

  const fetchAuthors = async () => {
    try {
      const response = await fetch('http://localhost:5000/hits'); // Replace with your backend API endpoint
      const data = await response.json();

      setTopHubs(data.best_hubs);
      setTopAuthorities(data.best_authors);
    } catch (error) {
      console.error('Error fetching authors:', error);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.header}>Top Hubs</h2>
      {topHubs.length > 0 ? (
        <ul style={styles.list}>
          {topHubs.map((author, index) => (
            <li key={index} style={styles.listItem}>
              <div style={styles.title}>{author}</div>
            </li>
          ))}
        </ul>
      ) : (
        <p>Loading top hubs...</p>
      )}

      <h2 style={styles.header}>Top Authorities</h2>
      {topAuthorities.length > 0 ? (
        <ul style={styles.list}>
          {topAuthorities.map((author, index) => (
            <li key={index} style={styles.listItem}>
              <div style={styles.title}>{author}</div>
            </li>
          ))}
        </ul>
      ) : (
        <p>Loading top authorities...</p>
      )}
    </div>
  );
};

export default AuthorList;
