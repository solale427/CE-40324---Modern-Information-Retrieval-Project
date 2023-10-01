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
  pageRank: {
    fontSize: '14px',
    color: '#666666',
  },
};

const PaperList = () => {
  const [papers, setPapers] = useState([]);

  useEffect(() => {
    fetchPapers();
  }, []);

  const fetchPapers = async () => {
    try {
      const response = await fetch('http://localhost:5000/page-rank'); // Replace with your backend API endpoint
      const data = await response.json();
      setPapers(data);
    } catch (error) {
      console.error('Error fetching papers:', error);
    }
  };

  const sortedPapers = papers.sort((a, b) => b.pageRank - a.pageRank);

  return (
    <div style={styles.container}>
      <h2 style={styles.header}>List of Papers with Best Page Rank</h2>
      {papers.length > 0 ? (
        <ul>
          {sortedPapers.map(paper => (
            <li key={paper.id} style={styles.listItem}>
              <a href={paper.url}><div style={styles.title}>{paper.title}</div></a>
            </li>
          ))}
        </ul>
      ) : (
        <p>Loading papers...</p>
      )}
    </div>
  );
};

export default PaperList;
