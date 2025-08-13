import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { AIMessageChunk } from '@langchain/core/messages';
import dotenv from 'dotenv';
dotenv.config();

const model = new ChatGoogleGenerativeAI({
  model: 'gemini-2.5-flash-lite',
  maxOutputTokens: 2048,
  apiKey: process.env.GOOGLE_API_KEY
});

const response: AIMessageChunk = await model.invoke('hello, are you ready to help me');
console.log(response);
console.log('---');
console.log('Content:', response.content);
