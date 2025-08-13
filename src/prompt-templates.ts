import { ChatPromptTemplate, } from '@langchain/core/prompts';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import dotenv from 'dotenv';
dotenv.config();

const model = new ChatGoogleGenerativeAI({
  model: 'gemini-2.5-flash-lite',
  maxOutputTokens: 2048,
  apiKey: process.env.GOOGLE_API_KEY,
});

const prompt = ChatPromptTemplate.fromTemplate(
  'You are a helpful assistant. {input}'
);

//create chain
const chain = prompt.pipe(model);

//call chain
const response = await chain.invoke({ input: 'hello, are you ready to help me' });

console.log(response);