import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import App, { FormattedAnswer } from './App';

beforeAll(() => {
  window.matchMedia = window.matchMedia || (() => ({ matches: false }));
});

beforeEach(() => {
  localStorage.clear();
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

test('shows six plain-language questions for general readers and technical questions for researchers', () => {
  render(<App />);
  expect(screen.getAllByRole('button', { name: /\?/ })).toHaveLength(6);
  expect(screen.getByRole('button', { name: 'Where do password requests appear?' })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'What are the model performance confidence intervals?' })).not.toBeInTheDocument();

  fireEvent.change(screen.getByLabelText('Explain for'), { target: { value: 'researcher' } });
  expect(screen.getAllByRole('button', { name: /\?/ })).toHaveLength(13);
  expect(screen.getByRole('button', { name: 'What are the model performance confidence intervals?' })).toBeInTheDocument();
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
  });
  expect(screen.getByText('Where do password requests appear?', { selector: '.message.user p' })).toBeInTheDocument();
});
