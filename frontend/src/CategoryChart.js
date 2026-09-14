import React from 'react';
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

// Same tokens as App.css — kept in sync manually since SVG fills can't read CSS custom properties.
const COLORS = {
  light: { accent: '#2a78d6', sensitive: '#d03b3b', text: '#52514e', grid: '#e1e0d9' },
  dark: { accent: '#3987e5', sensitive: '#e66767', text: '#c3c2b7', grid: '#2c2c2a' },
};

function CustomTooltip({ active, payload, colors }) {
  if (!active || !payload || !payload.length) return null;
  const { category, count, sensitive } = payload[0].payload;
  return (
    <div
      style={{
        background: colors === COLORS.dark ? '#1a1a19' : '#fcfcfb',
        border: `1px solid ${colors.grid}`,
        borderRadius: 8,
        padding: '8px 12px',
        fontSize: '0.82rem',
        color: colors === COLORS.dark ? '#ffffff' : '#0b0b0b',
      }}
    >
      <div style={{ fontWeight: 600 }}>{category}</div>
      <div>{count} matched record{count === 1 ? '' : 's'}</div>
      {sensitive && (
        <div style={{ color: colors.sensitive, fontWeight: 600, marginTop: 2 }}>
          Sensitive category
        </div>
      )}
    </div>
  );
}

export default function CategoryChart({ data, theme }) {
  if (!data || data.length === 0) return null;

  const colors = theme === 'dark' ? COLORS.dark : COLORS.light;
  const hasSensitive = data.some((d) => d.sensitive);

  return (
    <div>
      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
            <XAxis
              dataKey="category"
              tick={{ fontSize: 11, fill: colors.text }}
              interval={0}
              angle={-20}
              textAnchor="end"
              height={60}
              stroke={colors.grid}
            />
            <YAxis
              allowDecimals={false}
              tick={{ fontSize: 11, fill: colors.text }}
              stroke={colors.grid}
              width={28}
            />
            <Tooltip content={<CustomTooltip colors={colors} />} cursor={{ fill: colors.grid, opacity: 0.4 }} />
            <Bar dataKey="count" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={600} animationEasing="ease-out">
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.sensitive ? colors.sensitive : colors.accent} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {hasSensitive && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.78rem', color: colors.text, marginTop: 4 }}>
          <span
            style={{
              display: 'inline-block',
              width: 10,
              height: 10,
              borderRadius: 3,
              background: colors.sensitive,
            }}
          />
          Sensitive category
        </div>
      )}
    </div>
  );
}
