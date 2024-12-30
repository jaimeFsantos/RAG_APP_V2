import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { TrendChart, ComparisonChart, PriceDistributionChart } from './chart-components';
import { determineVisualizationType, processMarketData, processFeatureImpact, processMarketSummary } from './data-processors';

const ChatVisualizer = ({ query, response, marketData }) => {
  const [visualizationType, setVisualizationType] = useState(null);
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    const type = determineVisualizationType(query);
    setVisualizationType(type);

    if (type) {
      let processedData;
      if (type === 'trend' || type === 'comparison' || type === 'price') {
        processedData = processMarketData(marketData, type);
      } else if (marketData?.feature_impact) {
        processedData = processFeatureImpact(marketData.feature_impact);
      } else if (marketData?.market_summary) {
        processedData = processMarketSummary(marketData.market_summary);
      }
      setChartData(processedData);
    }
  }, [query, marketData]);

  const renderVisualization = () => {
    if (!chartData || chartData.length === 0) return null;

    switch (visualizationType) {
      case 'trend':
        return <TrendChart data={chartData} />;
      case 'comparison':
        return <ComparisonChart data={chartData} />;
      case 'price':
        return <PriceDistributionChart data={chartData} />;
      default:
        return null;
    }
  };

  if (!visualizationType || !chartData) return null;

  return (
    <div className="w-full mt-4">
      <Card>
        <CardHeader>
          <CardTitle>
            {visualizationType === 'trend' && 'Market Trends Analysis'}
            {visualizationType === 'comparison' && 'Market Comparison'}
            {visualizationType === 'price' && 'Price Distribution'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {renderVisualization()}
        </CardContent>
      </Card>
    </div>
  );
};

export default ChatVisualizer;
