// tagging.service.ts
import { Injectable } from '@nestjs/common';
import axios from 'axios';
import FormData from 'form-data';
import * as fs from 'fs';

@Injectable()
export class TaggingService {
  async predictCategory(imagePath: string, title: string, description: string) {
    const form = new FormData();
    form.append('image', fs.createReadStream(imagePath));
    form.append('title', title);
    form.append('description', description);

    const res = await axios.post('http://localhost:8000/predict', form, {
      headers: form.getHeaders(),
    });

    return res.data;
  }
}
