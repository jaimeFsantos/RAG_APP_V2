import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ChatVisualizations = ({ queryType, data, insights }) => {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (data && Object.keys(data).length > 0) {
      preprocessData();
    }
  }, [data, queryType]);

  const preprocessData = () => {
    switch (queryType) {
      case 'price_trends':
        const trendData = Object.entries(data).map(([date, value]) => ({
          date,
          price: value
        }));
        setChartData(trendData);
        break;
      
      case 'feature_importance':
        const featureData = Object.entries(data).map(([feature, value]) => ({
          feature,
          importance: Math.abs(value)
        })).sort((a, b) => b.importance - a.importance).slice(0, 10);
        setChartData(featureData);
        break;
      
      case 'market_analysis':
        const marketData = Object.entries(data).map(([category, value]) => ({
          category,
          value
        }));
        setChartData(marketData);
        break;
        
      default:
        setChartData([]);
    }
  };

  const renderVisualization = () => {
    switch (queryType) {
      case 'price_trends':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="price" stroke="#2563eb" />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'feature_importance':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} layout="vertical">
              <XAxis type="number" />
              <YAxis dataKey="feature" type="category" width={150} />
              <Tooltip />
              <Legend />
              <Bar dataKey="importance" fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'market_analysis':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData}>
              <XAxis dataKey="category" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  if (!data || chartData.length === 0) return null;

  return (
    <Card className="w-full mt-4">
      <CardHeader>
        <CardTitle>
          {queryType === 'price_trends' && 'Price Trends Analysis'}
          {queryType === 'feature_importance' && 'Feature Importance Analysis'}
          {queryType === 'market_analysis' && 'Market Analysis'}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {renderVisualization()}
        {insights && (
          <div className="mt-4 text-sm text-gray-600">
            <h4 className="font-semibold mb-2">Key Insights:</h4>
            <ul className="list-disc pl-4">
              {insights.map((insight, index) => (
                <li key={index}>{insight}</li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ChatVisualizations;