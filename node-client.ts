// node-client.ts
import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';

async function send() {
  const form = new FormData();
  form.append('image', fs.createReadStream('/home/spandit/category-mapping-images/image-1.jpg'));
  form.append('title', 'Red cotton t-shirt');
  form.append('description', '100% cotton, round neck');

  const res = await axios.post('http://localhost:8000/predict', form, {
    headers: form.getHeaders(),
  });

  console.log(res.data);
}
send();
