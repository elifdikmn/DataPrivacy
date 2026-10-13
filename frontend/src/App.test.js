import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import App, { FormattedAnswer, resolveApiBase } from './App';

beforeAll(() => {
  window.matchMedia = window.matchMedia || (() => ({ matches: false }));
});

beforeEach(() => {
  localStorage.clear();
});

test('requires a configured backend URL in production and keeps a local development fallback', () => {
  expect(resolveApiBase('', 'production')).toBe('');
  expect(resolveApiBase('', 'development')).toBe('http://127.0.0.1:8000');
  expect(resolveApiBase(' https://api.example.com/ ', 'production')).toBe('https://api.example.com');
});

test('offers audience-specific answer modes', () => {
  render(<App />);
  const selector = screen.getByLabelText('Explain for');
  expect(selector).toHaveValue('general');
  expect(screen.getByRole('option', { name: 'General audience' })).toBeInTheDocument();
  expect(screen.queryByRole('option', { name: 'Student' })).not.toBeInTheDocument();
  expect(screen.getByRole('option', { name: 'Researcher' })).toBeInTheDocument();
});

test('renders model-selected emphasis as bold text', () => {
  render(<FormattedAnswer text="The key result is **7.3% sensitive data**." />);
  expect(screen.getByText('7.3% sensitive data').tagName).toBe('STRONG');
});

test('shows seven plain-language questions and four distinct researcher questions', () => {
  render(<App />);
  expect(screen.getAllByRole('button', { name: /\?/ })).toHaveLength(7);
  expect(screen.getByRole('button', { name: 'Where do password requests appear?' })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'What are the model performance confidence intervals?' })).not.toBeInTheDocument();

  fireEvent.change(screen.getByLabelText('Explain for'), { target: { value: 'researcher' } });
  expect(screen.getAllByRole('button', { name: /\?/ })).toHaveLength(11);
  expect(screen.getByRole('button', { name: 'What are the model performance confidence intervals?' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Do risky vs. safe clusters emerge, and which groups have the highest sensitive-data share?' })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Which plugin clusters have the highest sensitive-data share?' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Can the model suggest categories for "Other" records?' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Are there hidden sensitive parameters mislabeled as "Other"?' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: "How accurately can a parameter's category be predicted from its name and description?" })).toBeInTheDocument();
});

test('sends the canonical chart-mapped question while displaying its plain-language label', async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ answer: 'A short answer.', sources: [], chart_image: null }),
  });
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Where do password requests appear?' }));

  await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
  expect(JSON.parse(global.fetch.mock.calls[0][1].body)).toMatchObject({
    question: 'Which parameters collect passwords?',
    audience: 'general',
    history: [],
  });
  expect(screen.getByText('Where do password requests appear?', { selector: '.message.user p' })).toBeInTheDocument();
});

test('changing audience clears messages, draft input, and visible charts', async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ answer: 'An answer.', sources: [], chart_image: '/static/charts/example.png' }),
  });
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Where do password requests appear?' }));
  await screen.findByText('An answer.');
  fireEvent.click(screen.getByRole('button', { name: 'Show visualization' }));
  fireEvent.change(screen.getByPlaceholderText('Or type your own question…'), { target: { value: 'A draft' } });

  fireEvent.change(screen.getByLabelText('Explain for'), { target: { value: 'researcher' } });

  await waitFor(() => expect(screen.queryByText('An answer.')).not.toBeInTheDocument());
  expect(screen.queryByText('Where do password requests appear?', { selector: '.message.user p' })).not.toBeInTheDocument();
  expect(screen.getByPlaceholderText('Or type your own question…')).toHaveValue('');
  expect(screen.queryByRole('button', { name: 'Clear chat' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Hide visualization' })).not.toBeInTheDocument();
});

test('changing audience cancels a pending answer so the old chat cannot reappear', async () => {
  global.fetch = jest.fn().mockImplementation((_url, options) => new Promise((_resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
  }));
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Where do password requests appear?' }));
  const requestSignal = global.fetch.mock.calls[0][1].signal;

  fireEvent.change(screen.getByLabelText('Explain for'), { target: { value: 'researcher' } });

  expect(requestSignal.aborted).toBe(true);
  expect(screen.getByLabelText('Explain for')).toHaveValue('researcher');
  await waitFor(() => expect(screen.queryByText('Where do password requests appear?', { selector: '.message.user p' })).not.toBeInTheDocument());
  expect(screen.queryByText(/Error:/)).not.toBeInTheDocument();
});
