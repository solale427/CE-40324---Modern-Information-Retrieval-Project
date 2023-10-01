import React from 'react';
import SearchPage from './Paper';
import PaperList from "./PageRank";
import AuthorList from "./HITS";

const App = () => {
    return (
        <div>
            <SearchPage/>
            <PaperList/>
            <AuthorList/>
        </div>
    );
};

export default App;
