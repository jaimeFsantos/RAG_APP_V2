import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, BarChart, Bar, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

const ChatVisualizer = ({ query, response, marketData }) => {
  const [visualizationType, setVisualizationType] = useState(null);
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    determineVisualization(query, response);
  }, [query, response]);

  const determineVisualization = (query, response) => {
    // Identify visualization type based on query content
    const priceTerms = ['price', 'cost', 'value', 'expensive', 'cheap'];
    const trendTerms = ['trend', 'over time', 'historical', 'change'];
    const comparisonTerms = ['compare', 'difference', 'versus', 'vs'];
    
    const lowerQuery = query.toLowerCase();
    
    if (trendTerms.some(term => lowerQuery.includes(term))) {
      setVisualizationType('trend');
      prepareTrendData();
    } else if (comparisonTerms.some(term => lowerQuery.includes(term))) {
      setVisualizationType('comparison');
      prepareComparisonData();
    } else if (priceTerms.some(term => lowerQuery.includes(term))) {
      setVisualizationType('price');
      preparePriceData();
    }
  };

  const prepareTrendData = () => {
    // Process market data for trend visualization
    const trendData = marketData?.map(item => ({
      date: item.date,
      price: item.price,
      volume: item.volume
    })) || [];

    setChartData(trendData);
  };

  const prepareComparisonData = () => {
    // Process market data for comparison visualization
    const comparisonData = marketData?.map(item => ({
      category: item.category,
      value: item.value
    })) || [];

    setChartData(comparisonData);
  };

  const preparePriceData = () => {
    // Process market data for price distribution visualization
    const priceData = marketData?.map(item => ({
      range: item.priceRange,
      count: item.count
    })) || [];

    setChartData(priceData);
  };

  const renderVisualization = () => {
    switch (visualizationType) {
      case 'trend':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="price" stroke="#8884d8" />
              <Line type="monotone" dataKey="volume" stroke="#82ca9d" />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'comparison':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'price':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="range" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  return (
    <div className="w-full space-y-4">
      {visualizationType && (
        <Card>
          <CardHeader>
            <CardTitle>
              {visualizationType === 'trend' && 'Market Trends'}
              {visualizationType === 'comparison' && 'Market Comparison'}
              {visualizationType === 'price' && 'Price Distribution'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {renderVisualization()}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ChatVisualizer;
